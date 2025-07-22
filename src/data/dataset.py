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
    each "expert" corresponds to a different temporal sampling rate within
    multiple spatial resolutions. It returns a nested dictionary of clips.
    """
    def __init__(self, data_dir, num_frames=16, temporal_sampling_rates=(1, 2, 4, 8),
                 spatial_resolutions=((224, 224), (112, 112)), is_train=True):
        """
        Args:
            data_dir (str or Path): Path to the directory containing the dataset splits
                                    (e.g., 'data/unified_dataset/train').
            num_frames (int): The number of frames to sample for each temporal expert clip.
            temporal_sampling_rates (tuple of int): A tuple of clip steps, where each
                                                    step defines a temporal expert.
            spatial_resolutions (tuple of tuple): A tuple of (height, width) tuples for
                                                  each spatial expert resolution.
            is_train (bool): If True, applies training data augmentation.
                             Otherwise, uses validation/testing transformations.
        """

        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.temporal_sampling_rates = temporal_sampling_rates
        self.spatial_resolutions = spatial_resolutions
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
        for ext in ('*.mp4', '*.avi', '*.mov', '*.mkv'): # Added more common video extensions
            # Use rglob to find files in subdirectories
            video_files.extend(list(self.data_dir.rglob(ext)))
        if not video_files:
            raise FileNotFoundError(f"No videos found in {self.data_dir}")
        return video_files

    def _find_classes(self):
        """Finds the class folders in the dataset and creates a mapping."""
        classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        if not classes:
            raise FileNotFoundError(f"Couldn\'t find any class folder in {self.data_dir}.")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}
        return class_to_idx, idx_to_class

    def __len__(self):
        """Returns the total number of videos in the dataset."""
        return len(self.video_files)

    def __getitem__(self, idx):
        """
        Retrieves one video and returns a nested dictionary of preprocessed clips,
        one for each spatial resolution, each containing clips for different
        temporal sampling rates, plus the action label.
        Includes error handling for corrupt video files.
        """
        video_path = self.video_files[idx]
        action_name = video_path.parent.name
        label = self.class_to_idx[action_name]

        try:
            # Use decord for efficient video reading
            vr = VideoReader(str(video_path), ctx=cpu(0))
            total_frames = len(vr)

            if total_frames <= 0:
                raise ValueError(f"Video {video_path} has 0 or negative frames.")

            all_resolution_clips = {}

            # --- Determine consistent spatial augmentations for all spatial experts ---
            # To do this, we sample a single frame to get the parameters for the crop, etc.
            # This ensures the base crop/flip/color/rotation are the same, but applied
            # to different target output_sizes later.
            sample_frame_idx = min(0, total_frames - 1) # Ensure index is valid
            sample_frame = torch.from_numpy(vr.get_batch([sample_frame_idx]).asnumpy()).permute(0, 3, 1, 2).float() # (1, C, H_orig, W_orig)

            if self.is_train:
                # Decide if a flip will be applied consistently across all resolutions and temporal rates
                do_flip = random.random() > 0.5

                # Get consistent parameters for ColorJitter
                jitter_params = {
                    "brightness": random.uniform(0.5, 1.5), # Stronger
                    "contrast": random.uniform(0.5, 1.5),    # Stronger
                    "saturation": random.uniform(0.5, 1.5),  # Stronger
                    "hue": random.uniform(-0.2, 0.2),        # Stronger
                }
                # The order of applying jitter transforms is also randomized.
                jitter_order = [0, 1, 2, 3]
                random.shuffle(jitter_order)

                # Get a consistent angle for RandomRotation.
                rotation_angle = random.uniform(-30, 30) # Stronger
                # Decide if Gaussian Blur will be applied consistently across all resolutions and temporal rates
                do_gaussian_blur = random.random() > 0.5

            # --- Process the video for each spatial resolution expert ---
            for current_height, current_width in self.spatial_resolutions:
                current_output_size = (current_height, current_width)

                if self.is_train:
                    # Adjust scale for RandomResizedCrop based on resolution.
                    # Smaller output_size might imply 'zoomed out' view.
                    # Assuming 224x224 is a typical "normal" resolution for comparison.
                    if current_height < max(res[0] for res in self.spatial_resolutions): # if current_height is smaller than max height
                        crop_scale = (0.5, 1.0) # More aggressive "zoom out" or wider view
                    else:
                        crop_scale = (0.8, 1.0) # Typical "zoom in" for action focus

                    i, j, h, w = transforms.RandomResizedCrop.get_params(
                        sample_frame, scale=crop_scale, ratio=(0.75, 1.33)
                    )

                clips_for_resolution = {}
                # --- Process the clip for each temporal sampling expert within this resolution ---
                for rate in self.temporal_sampling_rates:
                    # Temporal sampling with random speed perturbation for training
                    if self.is_train:
                        # Perturb the effective duration of the clip to simulate speed changes
                        speed_perturbation_factor = random.uniform(0.8, 1.2) # e.g., +/- 20% speed variation
                        effective_clip_duration_frames = int(self.num_frames * rate * speed_perturbation_factor)

                        # Ensure effective_clip_duration_frames is at least self.num_frames
                        # to avoid issues with torch.linspace if start_index == end_index
                        effective_clip_duration_frames = max(self.num_frames, effective_clip_duration_frames)

                        start_index = random.randint(0, max(0, total_frames - effective_clip_duration_frames))
                    else:
                        # For validation/testing, use the original temporal sampling rate
                        effective_clip_duration_frames = self.num_frames * rate
                        start_index = max(0, (total_frames - effective_clip_duration_frames) // 2)

                    end_index = start_index + effective_clip_duration_frames
                    # Generate frame indices by linearly spacing over the (possibly perturbed) duration
                    frame_indices = torch.linspace(start_index, end_index, self.num_frames, dtype=torch.int).tolist()
                    frame_indices = [min(idx, total_frames - 1) for idx in frame_indices] # Ensure indices are within bounds

                    # Extract frames, convert to tensor, and scale to [0, 1]
                    frames_numpy = vr.get_batch(frame_indices).asnumpy()
                    clip = torch.from_numpy(frames_numpy).permute(0, 3, 1, 2).float() / 255.0  # F, C, H_orig, W_orig

                    # Apply the pre-determined spatial augmentations for this resolution
                    if self.is_train:
                        clip = F.resized_crop(clip, i, j, h, w, current_output_size, antialias=True)
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
                        if do_gaussian_blur:
                            # Apply GaussianBlur. Kernel size must be odd. Sigma chosen randomly.
                            clip = F.gaussian_blur(clip, kernel_size=(5, 5), sigma=(0.1, 2.0))
                        if random.random() < 0.15: # 15% probability to add Gaussian Noise
                            # Add Gaussian Noise (mean 0, std 0.01-0.05)
                            # Noise needs to be added before normalization but after scaling to [0,1]
                            noise = torch.randn_like(clip) * random.uniform(0.01, 0.05)
                            clip = torch.clamp(clip + noise, 0.0, 1.0) # Ensure values stay in [0,1]

                    else:
                        # For validation, we use a standard Resize and Center Crop
                        # Resize to slightly larger than output size then center crop
                        resize_size = int(current_output_size[0] * 1.1)
                        clip = F.resize(clip, resize_size, antialias=True)
                        clip = F.center_crop(clip, current_output_size)

                    # Apply Random Erasing to each frame within the clip (during training)
                    if self.is_train:
                        erased_frames = []
                        # Iterate through frames (clip is F, C, H, W)
                        for frame_idx in range(clip.shape[0]):
                            clip_frame = clip[frame_idx] # C, H, W
                            # Randomly decide whether to apply erasing to this frame
                            if random.random() < 0.25: # 25% probability to erase a frame
                                # Define erasing region parameters based on current output size
                                # Erase area 2-20% of frame area, aspect ratio 0.3-3.3
                                h_erase = int(current_output_size[0] * random.uniform(0.02, 0.2))
                                w_erase = int(current_output_size[1] * random.uniform(0.02, 0.2))

                                # Ensure h_erase and w_erase are not zero
                                h_erase = max(1, h_erase)
                                w_erase = max(1, w_erase)

                                # Random position for erasing
                                i_erase = random.randint(0, current_output_size[0] - h_erase)
                                j_erase = random.randint(0, current_output_size[1] - w_erase)

                                # Apply erase: fill with 0s (black)
                                clip_frame = F.erase(clip_frame, i_erase, j_erase, h_erase, w_erase, 0)
                            erased_frames.append(clip_frame)
                        clip = torch.stack(erased_frames) # Stack frames back to (F, C, H, W)

                    # Normalize the clip frame by frame and permute to (C, F, H, W)
                    normalized_clip = torch.stack([self.normalize(f) for f in clip])
                    clip = normalized_clip.permute(1, 0, 2, 3)

                    clips_for_resolution[f'clip_drop_{rate}'] = clip

                all_resolution_clips[f'res_{current_height}'] = clips_for_resolution

            return all_resolution_clips, label

        except Exception as e:
            print(f"Warning: Could not load or process video {video_path}. Skipping. Error: {e}")
            # Try to load a different video instead
            if len(self.video_files) > 1:
                # Ensure we pick a different video to avoid infinite loop on a single corrupt video
                new_idx = random.randint(0, len(self) - 1)
                if new_idx == idx: # If random picks the same index, try the next one
                    new_idx = (new_idx + 1) % len(self)
                return self.__getitem__(new_idx)
            else:
                raise RuntimeError(f"Only one video in the dataset, and it is corrupt: {video_path}. Cannot proceed. Error: {e}")
