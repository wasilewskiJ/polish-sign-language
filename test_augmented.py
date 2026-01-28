"""
Simple test script to verify augmented data loading works correctly.
Tests only class 'A' with 1 fold.
"""

import os
import sys
import numpy as np
from pathlib import Path
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.landmarks import extract_landmarks

def load_all_classes(raw_dir, augmented_dir):
    """Load all classes (original + augmented)"""
    print("\n1. Loading original samples for all classes...")
    
    X_original = []
    y_original = []
    
    classes = sorted([d.name for d in Path(raw_dir).iterdir() if d.is_dir()])
    print(f"   Found {len(classes)} classes: {', '.join(classes)}")
    
    for class_name in classes:
        class_dir = Path(raw_dir) / class_name
        image_files = sorted(class_dir.glob("*.jpg"))
        
        loaded = 0
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            landmarks = extract_landmarks(img)
            if landmarks is not None:
                X_original.append(landmarks)
                y_original.append(class_name)
                loaded += 1
        
        print(f"   {class_name}: {loaded} samples")
    
    print(f"   Total original: {len(X_original)} samples")
    
    print("\n2. Loading augmented samples for all classes...")
    
    X_augmented = []
    y_augmented = []
    
    for class_name in classes:
        aug_class_dir = Path(augmented_dir) / class_name
        if not aug_class_dir.exists():
            print(f"   Error: {class_name}: Augmented directory not found")
            continue
        
        aug_files = sorted(aug_class_dir.glob(f"{class_name}*_aug*.jpg"))
        
        loaded = 0
        for img_path in aug_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            landmarks = extract_landmarks(img)
            if landmarks is not None:
                X_augmented.append(landmarks)
                y_augmented.append(class_name)
                loaded += 1
        
        print(f"   {class_name}: {loaded} augmented samples")
    
    print(f"   Total augmented: {len(X_augmented)} samples")
    
    return (np.array(X_original), np.array(y_original), 
            np.array(X_augmented) if X_augmented else None, 
            np.array(y_augmented) if y_augmented else None)


def main():
    print("=" * 70)
    print("AUGMENTATION TEST - All Classes")
    print("=" * 70)
    
    raw_dir = "data/raw"
    augmented_dir = "data/augmented"
    
    X_orig, y_orig, X_aug, y_aug = load_all_classes(raw_dir, augmented_dir)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Original samples:   {len(X_orig)}")
    if X_aug is not None:
        print(f"Augmented samples:  {len(X_aug)}")
        print(f"Total samples:      {len(X_orig) + len(X_aug)}")
        print(f"\nExpected ratio: ~5x augmented per original")
        print(f"   Actual ratio:   {len(X_aug) / len(X_orig):.2f}x")
    else:
        print(f"Augmented samples:  0 (not found)")
        print(f"\nNo augmented data loaded!")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
