import os
import argparse
import yaml
from pathlib import Path
import cv2
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from torchvision import transforms
from torchvision.transforms import functional as F_vision # For resize and center_crop
from tqdm import tqdm # Import tqdm

# Assuming your project structure is project_xai/inference.py, project_xai/src/...
from src.data.dataset import VideoDataset
from src.models.model import MoEModel


def load_config(config_path):
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def perform_inference(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Data Loading setup for class mapping consistency ---
    # We need to initialize VideoDataset to get the class_to_idx mapping.
    # We don't need actual data loading, just the mapping.
    # Using a dummy data_dir and dataset_name just to initialize.
    # It's crucial that data_dir and dataset_name point to a valid dataset
    # that matches the one used for training to get correct class mapping.
    data_dir_for_mapping = cfg['data']['data_dir_for_mapping']
    dataset_name_for_mapping = cfg['data']['dataset_name_for_mapping']

    dummy_train_data_path = os.path.join(data_dir_for_mapping, dataset_name_for_mapping, 'train')

    # Ensure spatial_resolutions are parsed as tuples for VideoDataset
    spatial_resolutions_tuples = tuple(tuple(res) for res in cfg['data']['spatial_resolutions'])

    # Initialize VideoDataset purely for class mapping
    temp_dataset = VideoDataset(
        data_dir=dummy_train_data_path,
        num_frames=cfg['data']['num_frames'],
        temporal_sampling_rates=tuple(cfg['data']['temporal_sampling_rates']),
        spatial_resolutions=spatial_resolutions_tuples,
        is_train=False # Use validation/inference transforms for dummy dataset init
    )
    num_classes = len(temp_dataset.class_to_idx)
    idx_to_class = temp_dataset.idx_to_class
    print(f"Loaded class mapping for {num_classes} classes.")

    # --- Model Loading ---
    model_path = cfg['inference']['model_path']
    model = MoEModel(
        num_classes=num_classes,
        temporal_sampling_rates=tuple(cfg['data']['temporal_sampling_rates']),
        spatial_resolutions=spatial_resolutions_tuples,
        dropout_prob=cfg['model']['dropout_prob'] # Must match trained model's dropout
    ).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")

    # --- Video Processing and Inference ---
    video_path = Path(cfg['inference']['video_path'])
    output_path = Path(cfg['inference']['output_path'])
    output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found at: {video_path}")

    vr = VideoReader(str(video_path), ctx=cpu(0))
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    print(f"Input video: {video_path}, Total frames: {total_frames}, FPS: {fps:.2f}")

    # VideoWriter setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
    output_height, output_width = cfg['data']['spatial_resolutions'][0] # Use highest resolution for output video
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))
    if not out.isOpened():
        raise IOError(f"Could not open video writer for path: {output_path}")
    print(f"Writing annotated video to {output_path}")

    clip_length_in_frames_max = max(cfg['data']['temporal_sampling_rates']) * cfg['data']['num_frames']

    # Iterate through the video using a sliding window for inference
    # Overlap can be adjusted if needed, currently set to half the max clip length
    stride_frames = max(1, clip_length_in_frames_max // 2)

    for i in tqdm(range(0, total_frames - clip_length_in_frames_max + 1, stride_frames), desc="Inferencing"):
        expert_clips_nested = {}

        for current_height, current_width in spatial_resolutions_tuples:
            clips_for_resolution = {}
            for rate in cfg['data']['temporal_sampling_rates']:
                # Temporal sampling logic (center crop for inference)
                start_index = i
                effective_clip_duration_frames = cfg['data']['num_frames'] * rate
                end_index = start_index + effective_clip_duration_frames

                frame_indices = torch.linspace(start_index, end_index, cfg['data']['num_frames'], dtype=torch.int).tolist()
                frame_indices = [min(idx, total_frames - 1) for idx in frame_indices]

                # Extract frames and preprocess
                frames_numpy = vr.get_batch(frame_indices).asnumpy()
                clip = torch.from_numpy(frames_numpy).permute(0, 3, 1, 2).float() / 255.0 # F, C, H_orig, W_orig

                # Apply validation transforms: Resize and Center Crop
                resize_size = int(current_height * 1.1)
                clip = F_vision.resize(clip, resize_size, antialias=True)
                clip = F_vision.center_crop(clip, (current_height, current_width))

                # Normalize and permute to (C, F, H, W)
                normalized_clip = torch.stack([temp_dataset.normalize(f) for f in clip])
                clip = normalized_clip.permute(1, 0, 2, 3)

                clips_for_resolution[f'clip_drop_{rate}'] = clip
            expert_clips_nested[f'res_{current_height}'] = clips_for_resolution

        # Add batch dimension
        expert_clips_nested_batch = {
            res_key: {clip_key: clip_tensor.unsqueeze(0) for clip_key, clip_tensor in res_dict.items()}
            for res_key, res_dict in expert_clips_nested.items()
        }

        with torch.no_grad():
            logits, _, _ = model(expert_clips_nested_batch) # Gating weights not needed for inference

        predicted_class_idx = torch.argmax(logits, dim=1).item()
        predicted_action = idx_to_class[predicted_class_idx]

        # Read frames for actual output video annotation (use original resolution for clarity)
        # Only read frames from the current stride for annotation
        current_segment_frames_indices = list(range(i, min(i + stride_frames, total_frames)))
        display_frames = vr.get_batch(current_segment_frames_indices).asnumpy() # F, H, W, C

        for frame_np in display_frames:
            # Convert to OpenCV BGR format
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

            # Resize frame to output_height, output_width
            frame_bgr_resized = cv2.resize(frame_bgr, (output_width, output_height))

            # Overlay prediction text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            font_thickness = 2
            text_color = (0, 255, 0) # Green
            text_position = (10, 30) # Top-left

            cv2.putText(frame_bgr_resized, predicted_action, text_position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            out.write(frame_bgr_resized)

    out.release()
    print("Inference complete and annotated video saved.")


def main():
    parser = argparse.ArgumentParser(description="Two-Tier MoE Action Recognition Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    # Allow command-line overrides for paths
    parser.add_argument("--model_path", type=str, default=None, help="Override model_path in config.")
    parser.add_argument("--video_path", type=str, default=None, help="Override video_path in config.")
    parser.add_argument("--output_path", type=str, default=None, help="Override output_path in config.")

    args = parser.parse_args()

    cfg = load_config(args.config)

    # Apply command-line overrides
    if args.model_path is not None:
        cfg['inference']['model_path'] = args.model_path
    if args.video_path is not None:
        cfg['inference']['video_path'] = args.video_path
    if args.output_path is not None:
        cfg['inference']['output_path'] = args.output_path

    # Check for essential parameters after loading config and applying overrides
    if not cfg['inference'].get('model_path'):
        raise ValueError("model_path must be specified in config or via command-line.")
    if not cfg['inference'].get('video_path'):
        raise ValueError("video_path must be specified in config or via command-line.")
    if not cfg['inference'].get('output_path'):
        raise ValueError("output_path must be specified in config or via command-line.")
    # Also ensure data_dir_for_mapping and dataset_name_for_mapping are present
    if not cfg['data'].get('data_dir_for_mapping') or not cfg['data'].get('dataset_name_for_mapping'):
        raise ValueError("data_dir_for_mapping and dataset_name_for_mapping must be specified in config for class mapping.")

    perform_inference(cfg)

if __name__ == "__main__":
    main()
