import os
import shutil
import random
from pathlib import Path
import argparse

def organize_jhmdb_dataset(source_dir, target_dir, split_ratios=(0.7, 0.15, 0.15), dry_run=False):
    """
    Reorganizes the JHMDB dataset from a flat structure of action folders
    into a structured train/val/test format.

    The original directory structure is expected to be:
    JHMDB/
    ├── brush_hair/
    │   ├── video_name_1.avi
    │   └── ...
    ├── catch/
    └── ...

    The new structure will be:
    JHMDB_organized/
    ├── train/
    │   ├── brush_hair/
    │   └── ...
    ├── val/
    │   ├── brush_hair/
    │   └── ...
    └── test/
        ├── brush_hair/
        └── ...

    Args:
        source_dir (str or Path): The path to the original JHMDB dataset.
        target_dir (str or Path): The path where the organized dataset will be stored.
        split_ratios (tuple): A tuple with train, val, and test split ratios.
        dry_run (bool): If True, prints actions without moving files.
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    if not source_path.is_dir():
        print(f"Error: Source directory '{source_path}' not found.")
        return

    if target_path.exists() and not dry_run:
        print(f"Target directory '{target_path}' already exists. Removing it for a fresh start.")
        shutil.rmtree(target_path)

    # Create base train, val, and test directories
    train_dir = target_path / 'train'
    val_dir = target_path / 'val'
    test_dir = target_path / 'test'

    print("Creating new directory structure...")
    if not dry_run:
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
    print(f"  - {train_dir}")
    print(f"  - {val_dir}")
    print(f"  - {test_dir}")

    action_classes = sorted([d for d in source_path.iterdir() if d.is_dir()])
    print(f"\nFound {len(action_classes)} action classes.")

    for action_class_path in action_classes:
        action_name = action_class_path.name
        print(f"\nProcessing action: '{action_name}'")

        # Create subdirectories for the action
        if not dry_run:
            (train_dir / action_name).mkdir(exist_ok=True)
            (val_dir / action_name).mkdir(exist_ok=True)
            (test_dir / action_name).mkdir(exist_ok=True)

        # Get all video files and shuffle for random splitting
        videos = sorted(list(action_class_path.glob('*.avi')))
        random.shuffle(videos)

        # Calculate split indices
        num_videos = len(videos)
        train_end = int(num_videos * split_ratios[0])
        val_end = train_end + int(num_videos * split_ratios[1])

        # Split the video list
        train_videos = videos[:train_end]
        val_videos = videos[train_end:val_end]
        test_videos = videos[val_end:]

        print(f"  - Total videos: {num_videos}")
        print(f"  - Splitting into: {len(train_videos)} train, {len(val_videos)} val, {len(test_videos)} test")

        # Function to copy files to their new destinations
        def copy_files(file_list, dest_folder):
            if dry_run:
                return
            for video_path in file_list:
                shutil.copy(video_path, dest_folder / video_path.name)

        copy_files(train_videos, train_dir / action_name)
        copy_files(val_videos, val_dir / action_name)
        copy_files(test_videos, test_dir / action_name)

    print("\n-------------------------------------------")
    print(f"Dataset reorganization {'simulation' if dry_run else 'complete'}.")
    print(f"Organized dataset is located at: {target_path}")
    print("-------------------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Reorganize the JHMDB video dataset into train/val/test splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--source_dir',
        type=str,
        default='data/JHMDB',
        help='The source directory of the JHMDB dataset.'
    )
    parser.add_argument(
        '--target_dir',
        type=str,
        default='data/JHMDB_organized',
        help='The target directory for the reorganized dataset.'
    )
    parser.add_argument(
        '--split',
        nargs=3,
        type=float,
        default=[0.7, 0.15, 0.15],
        metavar=('TRAIN', 'VAL', 'TEST'),
        help='Train, validation, and test split ratios.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for shuffling to ensure reproducible splits.'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help="If set, script will only print actions without moving files."
    )

    args = parser.parse_args()

    # Ensure split ratios sum to 1.0
    if sum(args.split) != 1.0:
        raise ValueError(f"Split ratios must sum to 1.0, but got {sum(args.split)}")

    # Set the random seed for reproducibility
    random.seed(args.seed)

    organize_jhmdb_dataset(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        split_ratios=tuple(args.split),
        dry_run=args.dry_run
    )
