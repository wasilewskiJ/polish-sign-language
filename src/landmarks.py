# backend/translator/landmarks.py
from pathlib import Path
import os
import sys

import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

_detector = None

def _get_detector():
    global _detector
    if _detector is None:
        script_dir = Path(__file__).parent
        model_path = script_dir / "models" / "hand_landmarker.task"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Hand landmarker model not found at {model_path}")
        
        # Suppress C++ warnings during detector creation
        stderr_fd = sys.stderr.fileno()
        saved_stderr = os.dup(stderr_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd)
        
        try:
            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3
            )
            _detector = vision.HandLandmarker.create_from_options(options)
        finally:
            os.dup2(saved_stderr, stderr_fd)
            os.close(devnull)
            os.close(saved_stderr)
    
    return _detector


def extract_landmarks(frame: np.ndarray):
    detector = _get_detector()

    # Convert the frame into MediaPipe's obligatory format
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Detect hands (suppress C++ warnings)
    stderr_fd = sys.stderr.fileno()
    saved_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    
    try:
        detection_result = detector.detect(mp_image)
    except Exception as e:
        os.dup2(saved_stderr, stderr_fd)
        os.close(devnull)
        os.close(saved_stderr)
        print(f"Error during hand detection: {e}")
        return None
    finally:
        os.dup2(saved_stderr, stderr_fd)
        os.close(devnull)
        os.close(saved_stderr)

    # Check if hands were detected
    if not detection_result.hand_landmarks:
        return None

    return detection_result


def compute_landmark_relationships(detection_result) -> np.ndarray:
    """
    Compute relationships between hand landmarks (distances and angles).

    Args:
        detection_result: MediaPipe HandLandmarkerResult object containing hand landmarks.
                         Can also accept a list of 21 landmarks, each with (x, y, z) coordinates for backward compatibility.

    Returns:
        np.ndarray: 1D array of relationship features (distances and angles).

    Raises:
        ValueError: If no hands are detected or the number of landmarks is not 21.
    """
    # Handle both HandLandmarkerResult and list inputs
    if isinstance(detection_result, list):
        landmarks = detection_result
    else:
        # Check if detection_result is a HandLandmarkerResult object
        if (
            not hasattr(detection_result, "hand_landmarks")
            or not detection_result.hand_landmarks
        ):
            raise ValueError("No hands detected in the provided detection result")

        # Extract landmarks from the first detected hand
        hand_landmarks = detection_result.hand_landmarks[0]  # First detected hand
        landmarks = [
            [landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks
        ]

    # Ensure we have exactly 21 landmarks
    if len(landmarks) != 21:
        raise ValueError(f"Expected 21 landmarks, got {len(landmarks)}")

    # Convert landmarks to a NumPy array for easier computation
    landmarks_array = np.array(landmarks)  # Shape: (21, 3)

    # Compute distances (10 features)
    # - Wrist (landmark 0) to each fingertip (landmarks 4, 8, 12, 16, 20)
    # - Between adjacent fingertips (4, 8), (8, 12), (12, 16), (16, 20)
    # - Wrist to palm center (average of landmarks 5, 9, 13, 17)
    distances = []
    wrist = landmarks_array[0]  # Landmark 0 (wrist)
    fingertips = [landmarks_array[i] for i in [4, 8, 12, 16, 20]]  # Fingertips
    palm_center = np.mean(
        [landmarks_array[i] for i in [5, 9, 13, 17]], axis=0
    )  # Palm center

    # Wrist to fingertips (5 distances)
    for fingertip in fingertips:
        distance = np.sqrt(np.sum((wrist - fingertip) ** 2))
        distances.append(distance)

    # Between adjacent fingertips (4 distances)
    for i in range(len(fingertips) - 1):
        distance = np.sqrt(np.sum((fingertips[i] - fingertips[i + 1]) ** 2))
        distances.append(distance)

    # Wrist to palm center (1 distance)
    distance = np.sqrt(np.sum((wrist - palm_center) ** 2))
    distances.append(distance)

    # Compute angles (5 features)
    # - Angle at each finger's MCP joint (landmarks 2, 6, 10, 14, 18)
    # - Use wrist (landmark 0) and fingertip (landmarks 4, 8, 12, 16, 20) as the other points
    angles = []
    mcp_joints = [2, 6, 10, 14, 18]  # MCP joints for thumb, index, middle, ring, pinky
    fingertips_indices = [4, 8, 12, 16, 20]

    for mcp_idx, fingertip_idx in zip(mcp_joints, fingertips_indices, strict=False):
        mcp = landmarks_array[mcp_idx]  # MCP joint
        fingertip = landmarks_array[fingertip_idx]  # Fingertip
        # Vectors: MCP to wrist, MCP to fingertip
        vec1 = wrist - mcp
        vec2 = fingertip - mcp
        # Compute angle using dot product
        cos_angle = np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6
        )  # Avoid division by zero
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # Angle in radians
        angles.append(angle)

    # Combine distances and angles into a single feature vector
    relationships = np.array(distances + angles)  # Shape: (15,)
    return relationships



