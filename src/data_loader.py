"""Data loading and feature extraction."""
import sys
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

try:
    from .landmarks import extract_landmarks, compute_landmark_relationships
except ImportError:
    from landmarks import extract_landmarks, compute_landmark_relationships


def load_augmented_images_for_split(image_paths, augmented_dir, num_augmentations=5):
    augmented_dir = Path(augmented_dir)
    augmented_paths = []
    augmented_labels = []
    
    for img_path in image_paths:
        img_path = Path(img_path)
        letter = img_path.parent.name  # e.g., "K"
        base_name = img_path.stem  # e.g., "K21"
        
        for aug_idx in range(num_augmentations):
            aug_name = f"{base_name}_aug{aug_idx}.jpg"
            aug_path = augmented_dir / letter / aug_name
            
            if aug_path.exists():
                augmented_paths.append(aug_path)
                augmented_labels.append(letter)
    
    return np.array(augmented_paths), np.array(augmented_labels)


def load_augmented_features(raw_dir, augmented_dir):

    raw_path = Path(raw_dir)
    augmented_path = Path(augmented_dir)
    
    print(f"\nLoading augmented features for training set...")
    
    X_list = []
    y_list = []
    original_idx_list = []
    is_augmented_list = []
    
    idx = 0
    
    for letter_dir in sorted(raw_path.iterdir()):
        if not letter_dir.is_dir() or letter_dir.name.startswith('.'):
            continue
            
        letter = letter_dir.name
        aug_letter_dir = augmented_path / letter
        
        for img_path in sorted(letter_dir.glob("*.jpg")):
            img_name_no_ext = img_path.stem
            
            img = cv2.imread(str(img_path))
            if img is None:
                idx += 1
                continue
            
            try:
                detection_result = extract_landmarks(img)
                if not detection_result or not detection_result.hand_landmarks:
                    idx += 1
                    continue
                
                relationships = compute_landmark_relationships(detection_result)
                hand_landmarks = detection_result.hand_landmarks[0]
                raw_coords = np.array([
                    [lm.x, lm.y, lm.z] for lm in hand_landmarks
                ]).flatten()
                
                features = np.concatenate([raw_coords, relationships])
                X_list.append(features)
                y_list.append(letter)
                original_idx_list.append(idx)
                is_augmented_list.append(False)
                
                if aug_letter_dir.exists():
                    for aug_img_path in sorted(aug_letter_dir.glob(f"{img_name_no_ext}_aug*.jpg")):
                        aug_img = cv2.imread(str(aug_img_path))
                        if aug_img is None:
                            continue
                        
                        try:
                            aug_detection = extract_landmarks(aug_img)
                            if not aug_detection or not aug_detection.hand_landmarks:
                                continue
                            
                            aug_relationships = compute_landmark_relationships(aug_detection)
                            aug_hand_landmarks = aug_detection.hand_landmarks[0]
                            aug_raw_coords = np.array([
                                [lm.x, lm.y, lm.z] for lm in aug_hand_landmarks
                            ]).flatten()
                            
                            aug_features = np.concatenate([aug_raw_coords, aug_relationships])
                            X_list.append(aug_features)
                            y_list.append(letter)
                            original_idx_list.append(idx)
                            is_augmented_list.append(True)
                        except:
                            continue
                
                idx += 1
                
            except:
                idx += 1
                continue
    
    result = {
        'X': np.array(X_list),
        'y': np.array(y_list),
        'original_idx': np.array(original_idx_list),
        'is_augmented': np.array(is_augmented_list),
    }
    
    print(f"Loaded {len(result['X'])} samples")
    print(f"   - Original: {(~result['is_augmented']).sum()}")
    print(f"   - Augmented: {result['is_augmented'].sum()}")
    
    return result


def get_augmented_data_for_fold(augmented_data, train_indices, val_indices, use_augmentation=True):
    X_all = augmented_data['X']
    y_all = augmented_data['y']
    original_idx = augmented_data['original_idx']
    is_augmented = augmented_data['is_augmented']
    
    if use_augmentation:
        train_mask = np.isin(original_idx, train_indices)
    else:
        train_mask = np.isin(original_idx, train_indices) & (~is_augmented)
    
    val_mask = np.isin(original_idx, val_indices) & (~is_augmented)
    
    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    X_val = X_all[val_mask]
    y_val = y_all[val_mask]
    
    return X_train, y_train, X_val, y_val


def load_raw_data_and_extract_features(raw_dir):

    raw_path = Path(raw_dir)
    print(f"Loading data from {raw_path}...")
    
    X_list = []
    y_list = []
    skipped = {}
    
    for letter_dir in sorted(raw_path.iterdir()):
        if not letter_dir.is_dir() or letter_dir.name.startswith('.'):
            continue
            
        letter = letter_dir.name
        print(f"  Processing {letter}...", end=" ", flush=True)
        
        count = 0
        letter_skipped = 0
        
        for img_path in sorted(letter_dir.glob("*.jpg")):
            img = cv2.imread(str(img_path))
            if img is None:
                letter_skipped += 1
                continue
            
            try:
                detection_result = extract_landmarks(img)
                
                if not detection_result or not detection_result.hand_landmarks:
                    letter_skipped += 1
                    continue
                
                relationships = compute_landmark_relationships(detection_result)
                
                hand_landmarks = detection_result.hand_landmarks[0]
                raw_coords = np.array([
                    [lm.x, lm.y, lm.z] for lm in hand_landmarks
                ]).flatten()
                
                features = np.concatenate([raw_coords, relationships])
                
                X_list.append(features)
                y_list.append(letter)
                count += 1
                
            except Exception as e:
                print(f"\n    Error processing {img_path.name}: {e}")
                letter_skipped += 1
                continue
        
        print(f"{count} samples loaded", end="")
        if letter_skipped > 0:
            print(f" ({letter_skipped} skipped)", end="")
            skipped[letter] = letter_skipped
        print()
    
    if not X_list:
        raise ValueError("No data loaded! Check if raw directory has images.")
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"\nTotal: {len(X)} samples, {len(np.unique(y))} classes")
    
    if skipped:
        print(f"Skipped samples per class: {skipped}")
    
    class_counts = Counter(y)
    print("\nClass distribution:")
    for label in sorted(class_counts.keys()):
        print(f"  {label}: {class_counts[label]} samples")
    
    return X, y


def load_image_paths_and_labels(raw_dir):

    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise ValueError(f"Raw directory not found: {raw_dir}")
    
    all_files = []
    all_labels = []
    
    for label_dir in sorted(raw_dir.iterdir()):
        if label_dir.is_dir() and not label_dir.name.startswith('.'):
            label = label_dir.name
            files = sorted(label_dir.glob("*.jpg"))
            all_files.extend(files)
            all_labels.extend([label] * len(files))
    
    if not all_files:
        raise ValueError(f"No images found in {raw_dir}")
    
    return np.array(all_files), np.array(all_labels)
