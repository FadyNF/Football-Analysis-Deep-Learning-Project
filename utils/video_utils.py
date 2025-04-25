import cv2  # OpenCV library for video processing
import tqdm  # Progress bar library for visualizing processing progress


def read_video(path):
    """
    Read a video file and extract all frames into a list.

    Args:
        path (str): Path to the video file to be read

    Returns:
        list: List of frames (as numpy arrays) from the video
    """
    capture = cv2.VideoCapture(
        path
    )  # Create a VideoCapture object to read from the video file
    frames = []  # Initialize empty list to store video frames

    # Loop through all frames in the video
    while True:
        ret, frame = (
            capture.read()
        )  # Read next frame (ret is True if frame was read successfully)
        if not ret:  # If no more frames are available
            break
        frames.append(frame)  # Add the frame to our list

    return frames


def save_video(output_frames, output_path):
    """
    Save a sequence of frames as a video file.

    Args:
        output_frames (list): List of frames (numpy arrays) to save as video
        output_path (str): Path where the output video should be saved
    """
    # Define the codec to use for video compression (XVID - common for .avi format)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    # Create a VideoWriter object with:
    # - output path
    # - codec
    # - frame rate (24 FPS)
    # - frame dimensions taken from the first frame
    out = cv2.VideoWriter(
        output_path, fourcc, 24, (output_frames[0].shape[1], output_frames[0].shape[0])
    )

    # Create a progress bar to show saving progress
    pbar = tqdm.tqdm(total=len(output_frames), desc="Saving video", unit="frame")

    # Write each frame to the output video
    for frame in output_frames:
        out.write(frame)
        pbar.update(1)  # Update progress bar

    # Clean up resources
    pbar.close()  # Close progress bar
    out.release()  # Release the VideoWriter to finalize the video file
