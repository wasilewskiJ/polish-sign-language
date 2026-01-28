from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

try:
    from .landmarks import compute_landmark_relationships, extract_landmarks
except ImportError:
    from landmarks import compute_landmark_relationships, extract_landmarks


def augment_image(image):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    # Random parameters for augmentation
    theta_deg = np.random.uniform(-10, 10)  # Rotation angle in degrees
    theta_rad = np.deg2rad(theta_deg)
    scale = np.random.uniform(0.9, 1.1)  # Scaling factor
    tx = np.random.uniform(-0.1 * w, 0.1 * w)  # Translation in x
    ty = np.random.uniform(-0.1 * h, 0.1 * h)  # Translation in y
    brightness_factor = np.random.uniform(0.8, 1.2)  # Brightness adjustment

    # Scaling matrix around center
    T_scale = np.array(
        [
            [scale, 0, center[0] * (1 - scale)],
            [0, scale, center[1] * (1 - scale)],
            [0, 0, 1],
        ]
    )

    # Rotation matrix around center
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    T_rotate = np.array(
        [
            [
                cos_theta,
                -sin_theta,
                center[0] * (1 - cos_theta) + center[1] * sin_theta,
            ],
            [sin_theta, cos_theta, center[1] * (1 - cos_theta) - center[0] * sin_theta],
            [0, 0, 1],
        ]
    )

    # Translation matrix
    T_translate = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

    # Combine transformations: T_translate * T_rotate * T_scale
    M_total = T_translate @ T_rotate @ T_scale
    affine_matrix = M_total[:2, :]  # Extract 2x3 affine matrix

    # Apply affine transformation
    aug_img = cv2.warpAffine(
        image,
        affine_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Adjust brightness
    aug_img = aug_img.astype(np.float32) * brightness_factor
    aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)

    return aug_img


def process_dataset(
    raw_data_dir="data/raw", output_base_dir="data", num_augmentations=0
):
    # Resolve the project root (pjm2jp/) by navigating up from this script's location
    project_root = Path(__file__).parent

    # Resolve raw_data_dir and output_base_dir relative to the project root
    raw_data_path = project_root / raw_data_dir
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw data directory {raw_data_path} does not exist.")

    train_dir = project_root / output_base_dir / "train"
    val_dir = project_root / output_base_dir / "val"
    test_dir = project_root / output_base_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    total_images_processed = 0

    # Process each letter folder (A, B, C, etc.)
    letter_dirs = [p for p in raw_data_path.iterdir() if p.is_dir()]
    letter_counts = {}
    for letter_dir in letter_dirs:
        letter_counts[letter_dir] = sum(1 for _ in letter_dir.glob("*.jpg"))

    if not letter_counts:
        print("No letter directories found. Nothing to process.")
        return

    min_count = min(count for count in letter_counts.values() if count > 0)
    if min_count == 0:
        print("At least one letter has zero images. Nothing to process.")
        return

    for letter_dir in letter_dirs:
        if not letter_dir.is_dir():
            continue

        letter = letter_dir.name  # e.g., "A"
        print(f"Processing letter {letter}...")

        features_list = []
        labels = []

        img_paths = sorted(letter_dir.glob("*.jpg"))
        img_paths = img_paths[:min_count]
        for img_path in img_paths:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not load image {img_path}")
                continue

            # Process original image
            try:
                detection_result = extract_landmarks(img)
                if not detection_result or not detection_result.hand_landmarks:
                    print(f"No hands detected in original {img_path}")
                    continue
                relationships = compute_landmark_relationships(detection_result)
                hand_landmarks = detection_result.hand_landmarks[0]
                raw_coordinates = np.array(
                    [
                        [landmark.x, landmark.y, landmark.z]
                        for landmark in hand_landmarks
                    ]
                ).flatten()
                features = np.concatenate([raw_coordinates, relationships])
                features_list.append(features)
                labels.append(letter)
            except Exception as e:
                print(f"Error processing original image {img_path}: {e}")
                continue

            # Process augmented versions
            for _ in range(num_augmentations):
                aug_img = augment_image(img)
                try:
                    detection_result = extract_landmarks(aug_img)
                    if not detection_result or not detection_result.hand_landmarks:
                        continue  # Skip if no hands detected in augmented image
                    relationships = compute_landmark_relationships(detection_result)
                    hand_landmarks = detection_result.hand_landmarks[0]
                    raw_coordinates = np.array(
                        [
                            [landmark.x, landmark.y, landmark.z]
                            for landmark in hand_landmarks
                        ]
                    ).flatten()
                    features = np.concatenate([raw_coordinates, relationships])
                    features_list.append(features)
                    labels.append(letter)
                except Exception as e:
                    print(f"Error processing augmented image for {img_path}: {e}")
                    continue

        if not features_list:
            print(f"No valid data for letter {letter}. Skipping...")
            continue

        # Convert to NumPy arrays
        features_array = np.array(features_list)
        labels_array = np.array(labels)

        # Split into train/val/test (80-10-10)
        train_features, temp_features, train_labels, temp_labels = train_test_split(
            features_array, labels_array, test_size=0.2, random_state=50
        )
        val_features, test_features, val_labels, test_labels = train_test_split(
            temp_features, temp_labels, test_size=0.5, random_state=50
        )

        # Save processed data as NumPy arrays
        np.save(train_dir / f"features_{letter}.npy", train_features)
        np.save(train_dir / f"labels_{letter}.npy", train_labels)
        np.save(val_dir / f"features_{letter}.npy", val_features)
        np.save(val_dir / f"labels_{letter}.npy", val_labels)
        np.save(test_dir / f"features_{letter}.npy", test_features)
        np.save(test_dir / f"labels_{letter}.npy", test_labels)

        total_images_processed += len(features_list) // (
            num_augmentations + 1
        )  # Count only original images
        print(
            f"Processed {len(features_list)} feature vectors (including {len(features_list) - total_images_processed} augmented) for letter {letter}."
        )

    print(
        f"Dataset processing completed. Total original images processed: {total_images_processed}"
    )


if __name__ == "__main__":
    process_dataset()
