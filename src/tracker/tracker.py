from ultralytics import YOLO  # Import YOLO model for object detection
import supervision as sv  # Import supervision for tracking capabilities
import pickle  # For serializing and deserializing Python objects
import os  # For file and path operations
import sys  # For system-specific parameters and functions
import cv2  # OpenCV for computer vision tasks
import numpy as np  # NumPy for numerical operations

# Add parent directory to path to import local modules
sys.path.append("../")
from utils import (
    get_bbox_center,
    get_bbox_width,
)  # Import utility functions for bounding box operations


class Tracker:
    """
    A class for tracking objects (players, referees, goalkeepers, and the ball) in football videos
    using YOLO for detection and ByteTrack for tracking.
    """

    def __init__(self, model_path):
        """
        Initialize the tracker with a YOLO model and ByteTrack tracker.

        Args:
            model_path (str): Path to the pre-trained YOLO model
        """
        self.model = YOLO(model_path)  # Load YOLO model
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,  # Confidence threshold to activate new tracks
            minimum_matching_threshold=0.9,  # Minimum IoU threshold for matching tracks
        )

    def detect_frames(self, frames):
        """
        Process frames in batches to detect objects using YOLO.

        Args:
            frames (list): List of video frames to process

        Returns:
            list: YOLO detection results for each frame
        """
        batch_size = 20  # Number of frames to process at once
        detections = []

        # Process frames in batches to improve performance
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.track(
                frames[i : i + batch_size], conf=0.1
            )  # Detect and track with low confidence threshold
            detections += detections_batch

        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Track objects in video frames or load tracking data from a saved file.

        Args:
            frames (list): List of video frames
            read_from_stub (bool): Whether to read tracking data from a saved file
            stub_path (str): Path to the saved tracking data file

        Returns:
            dict: Dictionary containing tracking data for players, goalkeepers, referees, and ball
        """
        # If stub file exists and read_from_stub is enabled, load tracking data from file
        if (
            read_from_stub == True
            and stub_path is not None
            and os.path.exists(stub_path)
        ):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks

        # Otherwise, perform detection on frames
        detections = self.detect_frames(frames)

        # Initialize tracking dictionary for different object types
        tracks = {"players": [], "goalkeepers": [], "referees": [], "ball": []}

        # Process each frame's detections
        for frame_num, detection in enumerate(detections):
            # Get class names mapping from detection model
            class_names = detection.names
            class_names_inversed = {
                v: k for k, v in class_names.items()
            }  # Invert mapping for lookup by name

            # Convert YOLO detections to supervision format
            detections_sv = sv.Detections.from_ultralytics(detection)

            # Update tracks with new detections
            detections_with_tracks = self.tracker.update_with_detections(detections_sv)

            # Initialize empty dictionaries for each type of object in the current frame
            tracks["players"].append({})
            tracks["goalkeepers"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # Process tracked detections (players, referees, goalkeepers)
            for frame_detection in detections_with_tracks:
                bbox = frame_detection[0].tolist()  # Bounding box coordinates
                class_id = frame_detection[3]  # Class ID (player, referee, etc.)
                track_id = frame_detection[4]  # Unique tracking ID

                # Store player detections
                if class_id == class_names_inversed["player"]:
                    tracks["players"][frame_num][track_id] = {
                        "bbox": bbox,
                        "confidence": frame_detection[2],
                    }

                # Store referee detections
                if class_id == class_names_inversed["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

                # Store goalkeeper detections
                if class_id == class_names_inversed["goalkeeper"]:
                    tracks["goalkeepers"][frame_num][track_id] = {"bbox": bbox}

            # Process ball detections separately (not tracked by ByteTrack)
            for frame_detection in detections_sv:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                # Store ball detections (always assigned ID 1)
                if class_id == class_names_inversed["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        # Save tracking data to file if stub_path is provided
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)
        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        """
        Draw an ellipse at the bottom of a player's bounding box to represent their position.
        Optionally adds track ID in a rectangle.

        Args:
            frame (ndarray): Video frame to draw on
            bbox (list): Bounding box coordinates [x1, y1, x2, y2]
            color (tuple): RGB color for the ellipse
            track_id (int, optional): Track ID to display

        Returns:
            ndarray: Frame with drawn ellipse
        """
        y2 = int(bbox[3])  # Bottom of bounding box

        # Get center point and width of bounding box
        x_center, _ = get_bbox_center(bbox)
        width = get_bbox_width(bbox)

        # Draw ellipse at the bottom of the bounding box
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(
                int(width),
                int(0.35 * width),
            ),  # Ellipse dimensions based on bbox width
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        # Draw rectangle with track ID if provided
        if track_id is not None:
            # Rectangle dimensions and position for the ID display
            rectangle_width = 40
            rectangle_height = 20
            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2
            y1_rect = (y2 - rectangle_height // 2) + 15
            y2_rect = (y2 + rectangle_height // 2) + 15

            # Draw filled rectangle for track ID
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED,
            )

            # Adjust text position based on track ID length
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            # Add track ID text
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),  # Black text
                2,
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        """
        Draw a triangle above the ball's bounding box.

        Args:
            frame (ndarray): Video frame to draw on
            bbox (list): Bounding box coordinates [x1, y1, x2, y2]
            color (tuple): RGB color for the triangle

        Returns:
            ndarray: Frame with drawn triangle
        """
        y = int(bbox[1])  # Top of bounding box
        x, _ = get_bbox_center(bbox)  # Center x-coordinate

        # Define triangle points
        triangle_points = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])

        # Draw filled triangle
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        # Draw triangle outline
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_annotations(self, video_frames, tracks):
        """
        Draw visual annotations for all tracked objects on video frames.

        Args:
            video_frames (list): List of video frames
            tracks (dict): Dictionary containing tracking data

        Returns:
            list: Frames with annotations drawn
        """
        output_video_frames = []

        # Process each frame
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()  # Create a copy to avoid modifying the original

            # Get tracking data for current frame
            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            goalie_dict = tracks["goalkeepers"][frame_num]

            # Draw players
            for track_id, player in player_dict.items():
                color = player.get(
                    "team_color", (0, 0, 255)
                )  # Use team color if assigned, otherwise red
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

            # Draw referees (yellow)
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Draw goalkeepers (cyan)
            for _, goali in goalie_dict.items():
                frame = self.draw_ellipse(frame, goali["bbox"], (255, 255, 0))

            # Draw ball (blue triangle)
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (255, 0, 0))

            output_video_frames.append(frame)

        return output_video_frames
