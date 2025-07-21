import random
from pathlib import Path

import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision import transforms


class VideoDataset(Dataset):
    """
    PyTorch Dataset for loading and preprocessing video clips for action recognition.

    This implementation is tailored for a Mixture-of-Experts (MoE) model where
    each "expert" corresponds to a different temporal sampling rate. It returns a
    dictionary of clips, each sampled at a different rate.
    """
    def __init__(self, data_dir, num_frames=16, temporal_sampling_rates=(1, 2, 4, 8),
                 height=224, width=224, is_train=True):
        """
        Args:
            data_dir (str or Path): Path to the directory containing the dataset splits
                                    (e.g., 'data/KARD_video_organized/train').
            num_frames (int): The number of frames to sample for each expert clip.
            temporal_sampling_rates (tuple of int): A tuple of clip steps, where each
                                                    step defines an expert.
            height (int): The final height of the frames.
            width (int): The final width of the frames.
            is_train (bool): If True, applies training data augmentation.
                             Otherwise, uses validation/testing transformations.
        """
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.temporal_sampling_rates = temporal_sampling_rates
        self.output_size = (height, width)
        self.is_train = is_train

        self.video_files = self._get_video_files()
        self.class_to_idx, self.idx_to_class = self._find_classes()

        # Define the normalization transformation
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def _get_video_files(self):
        """Scans the data directory for video files and returns a list."""
        video_files = []
        for ext in ('*.mp4', '*.avi'):
            # Use rglob to find files in subdirectories
            video_files.extend(list(self.data_dir.rglob(ext)))
        if not video_files:
            raise FileNotFoundError(f"No videos found in {self.data_dir}")
        return video_files

    def _find_classes(self):
        """Finds the class folders in the dataset and creates a mapping."""
        classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {self.data_dir}.")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}
        return class_to_idx, idx_to_class

    def __len__(self):
        """Returns the total number of videos in the dataset."""
        return len(self.video_files)

    def __getitem__(self, idx):
        """
        Retrieves one video and returns a dictionary of preprocessed clips,
        one for each temporal sampling rate, plus the action label.
        """
        video_path = self.video_files[idx]
        action_name = video_path.parent.name
        label = self.class_to_idx[action_name]

        # Use decord for efficient video reading
        vr = VideoReader(str(video_path), ctx=cpu(0))
        total_frames = len(vr)

        # --- Determine consistent spatial augmentations for all experts ---
        # To do this, we sample a single frame to get the parameters for the crop.
        # This ensures the crop and flip are the same for all temporal rates.
        sample_frame = torch.from_numpy(vr.get_batch([0]).asnumpy()).permute(0, 3, 1, 2).float()

        if self.is_train:
            # Get parameters for a single random crop that will be applied to all experts
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                sample_frame, scale=(0.8, 1.0), ratio=(0.75, 1.33)
            )
            # Decide if a flip will be applied to all experts
            do_flip = random.random() > 0.5

            # Get consistent parameters for ColorJitter, to be applied manually for consistency.
            jitter_params = {
                "brightness": random.uniform(0.6, 1.4),
                "contrast": random.uniform(0.6, 1.4),
                "saturation": random.uniform(0.6, 1.4),
                "hue": random.uniform(-0.1, 0.1),
            }
            # The order of applying jitter transforms is also randomized.
            jitter_order = [0, 1, 2, 3]
            random.shuffle(jitter_order)

            # Get a consistent angle for RandomRotation.
            rotation_angle = random.uniform(-15, 15)

        expert_clips = {}
        # --- Process the video for each temporal sampling expert ---
        for rate in self.temporal_sampling_rates:
            clip_length_in_frames = self.num_frames * rate

            # Perform temporal sampling (random for train, center for val)
            if self.is_train:
                start_index = random.randint(0, max(0, total_frames - clip_length_in_frames))
            else:
                start_index = max(0, (total_frames - clip_length_in_frames) // 2)

            end_index = start_index + clip_length_in_frames
            frame_indices = torch.linspace(start_index, end_index, self.num_frames, dtype=torch.int).tolist()
            frame_indices = [min(idx, total_frames - 1) for idx in frame_indices]

            # Extract frames, convert to tensor, and scale to [0, 1]
            frames = vr.get_batch(frame_indices).asnumpy()
            clip = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # F, C, H, W

            # Apply the pre-determined spatial augmentations
            if self.is_train:
                clip = F.resized_crop(clip, i, j, h, w, self.output_size, antialias=True)
                if do_flip:
                    clip = F.hflip(clip)

                # Apply ColorJitter and Rotation with the pre-determined consistent parameters
                for fn_id in jitter_order:
                    if fn_id == 0:
                        clip = F.adjust_brightness(clip, jitter_params["brightness"])
                    elif fn_id == 1:
                        clip = F.adjust_contrast(clip, jitter_params["contrast"])
                    elif fn_id == 2:
                        clip = F.adjust_saturation(clip, jitter_params["saturation"])
                    elif fn_id == 3:
                        clip = F.adjust_hue(clip, jitter_params["hue"])

                clip = F.rotate(clip, rotation_angle, interpolation=transforms.InterpolationMode.BILINEAR)
            else:
                # For validation, we use a standard Resize and Center Crop
                resize_size = int(self.output_size[0] * 1.1)
                clip = F.resize(clip, resize_size, antialias=True)
                clip = F.center_crop(clip, self.output_size)

            # Normalize the clip frame by frame and permute to (C, F, H, W)
            normalized_clip = torch.stack([self.normalize(f) for f in clip])
            clip = normalized_clip.permute(1, 0, 2, 3)

            expert_clips[f'clip_drop_{rate}'] = clip

        return expert_clips, label
