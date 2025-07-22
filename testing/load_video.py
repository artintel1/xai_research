import cv2
import os

def extract_frames_with_skips(video_path, num_frames_to_extract=5, frames_to_skip=3):
    """
    Loads a video, extracts a specified number of frames, skipping a given
    number of intermediate frames between each extracted frame.

    Args:
        video_path (str): The path to the video file.
        num_frames_to_extract (int): The total number of frames to extract.
        frames_to_skip (int): The number of consecutive frames to skip
                              between each extracted frame.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    extracted_count = 0
    frame_idx = 0

    print(f"Attempting to extract {num_frames_to_extract} frames from '{video_path}' "
          f"by skipping {frames_to_skip} frames between each extraction.")

    while extracted_count < num_frames_to_extract:
        # Read the current frame
        ret, frame = cap.read()

        if not ret:
            print(f"Warning: Reached end of video or failed to read frame after {extracted_count} frames extracted.")
            break

        # This is the frame we want to extract
        print(f"Extracted Frame {extracted_count + 1} (original index: {frame_idx}): Shape {frame.shape}, Dtype {frame.dtype}")
        # You can add code here to save the frame, e.g.:
        # cv2.imwrite(f"extracted_frame_{extracted_count + 1}.png", frame)

        extracted_count += 1
        frame_idx += 1 # Increment index for the extracted frame itself

        # Skip the next 'frames_to_skip' frames if more frames are needed
        if extracted_count < num_frames_to_extract:
            for _ in range(frames_to_skip):
                ret_skip, _ = cap.read() # Read and discard
                if not ret_skip:
                    print(f"Warning: Could not skip all {frames_to_skip} frames. Reached end of video early.")
                    break
                frame_idx += 1 # Increment index for skipped frames

    cap.release()
    print(f"Finished extracting frames. Total extracted: {extracted_count}")

if __name__ == "__main__":
    # --- IMPORTANT: Replace "path/to/your/video.mp4" with the actual path to your video file ---
    # Example: video_file = "/home/diat/project_xai/my_video.mp4"
    video_file = "/home/diat/project_xai/outputs/annotated_output_video4.mp4" # <--- YOU MUST CHANGE THIS PATH

    # Make sure to install opencv-python if you haven't already:
    # pip install opencv-python

    extract_frames_with_skips(video_file, num_frames_to_extract=5, frames_to_skip=3)
