import torch
import torch.nn as nn
import torchvision.transforms.functional as F_vision
import torchvision.transforms as transforms
from decord import VideoReader, cpu
import cv2
import numpy as np
import os
import argparse
from pathlib import Path

# Assume these are available relative to project_xai
from src.models.model import MoEModel
from src.data.dataset import VideoDataset # To get class names

def preprocess_clip_for_inference(frames_tensor, output_size, normalize_transform):
    """
    Applies the same preprocessing steps as VideoDataset validation mode.
    Args:
        frames_tensor (torch.Tensor): Tensor of frames (F, C, H, W) in [0, 1].
        output_size (tuple): Target (height, width).
        normalize_transform (torchvision.transforms.Normalize): Normalization transform.
    Returns:
        torch.Tensor: Preprocessed clip (C, F, H, W).
    """
    # For validation, we use a standard Resize and Center Crop
    resize_size = int(output_size[0] * 1.1)
    clip = F_vision.resize(frames_tensor, resize_size, antialias=True)
    clip = F_vision.center_crop(clip, output_size)

    # Normalize the clip frame by frame and permute to (C, F, H, W)
    normalized_clip = torch.stack([normalize_transform(f) for f in clip])
    clip = normalized_clip.permute(1, 0, 2, 3)
    return clip

def main():
    parser = argparse.ArgumentParser(description="MoE Temporal Sampling Action Recognition Inference")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to the input video file for inference.")
    parser.add_argument("--output_path", type=str, default="output_video.mp4",
                        help="Path to save the output annotated video.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the root data directory (e.g., data/KARD_video_organized) to infer class names.")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of the dataset (e.g., KARD_video_organized) to infer class names.")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Number of frames per expert clip (should match training).")
    parser.add_argument("--sampling_rates", type=int, nargs='+', default=[1, 2, 4, 8],
                        help="Temporal sampling rates for experts (should match training).")
    parser.add_argument("--height", type=int, default=224, help="Frame height (should match training).")
    parser.add_argument("--width", type=int, default=224, help="Frame width (should match training).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for inference (cuda or cpu).")
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # Load class mapping from a dummy dataset instance
    # This requires a valid path to an actual dataset, even if we don't load data from it.
    # It just needs to exist to find classes.
    try:
        dummy_dataset_path = os.path.join(args.data_dir, args.dataset_name, 'train') # or 'val'
        dummy_dataset = VideoDataset(
            data_dir=dummy_dataset_path,
            num_frames=args.num_frames, # These don't strictly matter for class mapping
            temporal_sampling_rates=tuple(args.sampling_rates),
            height=args.height, width=args.width, is_train=False
        )
        idx_to_class = dummy_dataset.idx_to_class
        print(f"Loaded class mapping for {len(idx_to_class)} classes.")
    except Exception as e:
        print(f"Error loading class mapping from dataset: {e}. Please ensure --data_dir and --dataset_name are correct and point to a valid dataset structure.")
        print("Proceeding without class mapping. Predictions will be raw indices.")
        idx_to_class = {i: f"Class {i}" for i in range(1000)} # Placeholder if mapping fails

    # Initialize model and load weights
    num_classes = len(idx_to_class) # Assuming the loaded mapping is correct
    model = MoEModel(num_classes=num_classes, temporal_sampling_rates=tuple(args.sampling_rates)).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()
    print(f"Model loaded from {args.model_path}")

    # Define normalization transform (must match training/validation)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    output_size = (args.height, args.width)

    # Load input video
    try:
        vr = VideoReader(str(args.video_path), ctx=cpu(0))
    except Exception as e:
        print(f"Error loading video: {e}. Please check --video_path: {args.video_path}")
        return

    total_frames = len(vr)
    fps = vr.get_avg_fps()
    print(f"Input video: {args.video_path}, Total frames: {total_frames}, FPS: {fps:.2f}")

    # Setup VideoWriter
    # Codec: H.264 is good, but might require specific OpenCV builds (e.g., FFMPEG enabled).
    # XVID (DIVX) or MJPG are safer bets for broader compatibility.
    # Using 'mp4v' for .mp4 usually works with standard OpenCV builds
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'XVID' or 'MJPG'
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (args.width, args.height))

    if not out.isOpened():
        print(f"Error: Could not open video writer for {args.output_path}. Check codec compatibility or output path.")
        return

    print(f"Writing annotated video to {args.output_path}")

    # Determine inference window
    max_rate = max(args.sampling_rates)
    segment_length_frames = args.num_frames * max_rate # Length of the longest expert clip

    current_prediction_label = "Processing..." # Default label

    # Loop through the video in segments
    for i in range(0, total_frames, segment_length_frames):
        end_frame_segment = min(i + segment_length_frames, total_frames)
        # Ensure we have enough frames for at least the base rate clip (num_frames * rate 1)
        if end_frame_segment - i < args.num_frames:
             # If remaining frames are too few for even the shortest clip, break or process what's left
             print(f"Skipping segment starting at frame {i}: Not enough frames ({end_frame_segment - i}) for a minimal clip ({args.num_frames}).")
             break # Or handle partial segments if needed

        # --- Extract and preprocess expert clips for the current segment ---
        expert_clips_on_device = {}
        for rate in args.sampling_rates:
            clip_start_index = i # Start of the segment
            clip_length_for_rate = args.num_frames * rate

            # Adjust start_index if clip_length_for_rate exceeds segment bounds
            # This is key for consistent temporal sampling within a logical "segment"
            if clip_start_index + clip_length_for_rate > total_frames:
                # If the chosen rate goes beyond video end, adjust start to fit the clip
                clip_start_index = total_frames - clip_length_for_rate
                clip_start_index = max(0, clip_start_index) # Ensure it's not negative

            # Generate frame indices for this specific rate within the segment
            # We want num_frames *from this clip_start_index* with the given rate step
            frame_indices_for_rate = torch.linspace(
                clip_start_index,
                clip_start_index + clip_length_for_rate -1, # End index inclusive
                args.num_frames,
                dtype=torch.int
            ).tolist()
            # Ensure indices do not exceed total_frames - 1
            frame_indices_for_rate = [min(idx, total_frames - 1) for idx in frame_indices_for_rate]

            # Extract frames and preprocess
            frames_np = vr.get_batch(frame_indices_for_rate).asnumpy()
            clip_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0 # F, C, H, W

            # Preprocess to (C, F, H, W)
            processed_clip = preprocess_clip_for_inference(clip_tensor, output_size, normalize)

            expert_clips_on_device[f'clip_drop_{rate}'] = processed_clip.unsqueeze(0).to(args.device) # Add batch dim

        # --- Perform inference for the segment ---
        with torch.no_grad():
            logits, _ = model(expert_clips_on_device)
            predicted_idx = torch.argmax(logits, dim=1).item()
            current_prediction_label = idx_to_class.get(predicted_idx, f"Unknown Class {predicted_idx}")

        # --- Annotate and write frames for the current segment ---
        # Read the actual frames in the segment that was processed and annotate them
        frames_to_annotate_indices = list(range(i, end_frame_segment))
        if not frames_to_annotate_indices:
            continue # Skip if no frames in segment

        frames_to_annotate_np = vr.get_batch(frames_to_annotate_indices).asnumpy()

        for frame_np in frames_to_annotate_np:
            # Convert frame from RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

            # Resize frame to output dimensions if not already
            if frame_bgr.shape[0] != args.height or frame_bgr.shape[1] != args.width:
                frame_bgr = cv2.resize(frame_bgr, (args.width, args.height))

            # Add text overlay
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            font_thickness = 2
            text_color = (0, 255, 0) # Green color in BGR

            # Get text size for positioning
            (text_width, text_height), baseline = cv2.getTextSize(current_prediction_label, font, font_scale, font_thickness)

            # Position text (e.g., bottom-left corner with some padding)
            padding = 10
            org = (padding, args.height - padding) # Bottom-left

            cv2.putText(frame_bgr, current_prediction_label, org, font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            out.write(frame_bgr)

    # Release resources
    out.release()

    print("Inference complete and output video saved.")

if __name__ == "__main__":
    main()
