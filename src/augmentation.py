"""
Image augmentation module for PSL recognition.
"""

import cv2
import numpy as np


def augment_image(img, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    aug_img = img.copy()
    h, w = aug_img.shape[:2]
    
    # Random rotation (-15 to +15 degrees)
    angle = np.random.uniform(-15, 15)
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # Random brightness adjustment (-30 to +30)
    brightness = np.random.randint(-30, 30)
    aug_img = cv2.convertScaleAbs(aug_img, alpha=1, beta=brightness)
    
    # Random noise
    if np.random.random() > 0.5:
        noise = np.random.randn(h, w, 3) * 10
        aug_img = np.clip(aug_img + noise, 0, 255).astype(np.uint8)
    
    return aug_img


def create_augmented_versions(img, num_augmentations=5, base_seed=None):

    augmented = []
    for i in range(num_augmentations):
        seed = None if base_seed is None else base_seed + i
        aug_img = augment_image(img, seed=seed)
        augmented.append(aug_img)
    return augmented


def generate_all_augmentations(raw_dir, output_dir, n_augmented_per_sample=3):
    from pathlib import Path
    
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_original = 0
    total_augmented = 0
    
    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith('.'):
            continue
        
        class_name = class_dir.name
        output_class_dir = output_dir / class_name
        output_class_dir.mkdir(exist_ok=True)
        
        image_files = sorted(class_dir.glob("*.jpg"))
        print(f"Processing class {class_name}: {len(image_files)} images...")
        
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  Warning: Could not read {img_path}")
                continue
            
            total_original += 1
            base_name = img_path.stem
            
            # Generate augmented versions
            augmented_imgs = create_augmented_versions(
                img, 
                num_augmentations=n_augmented_per_sample,
                base_seed=hash(base_name) % 10000
            )
            
            # Save augmented images
            for i, aug_img in enumerate(augmented_imgs):
                aug_filename = f"{base_name}_aug{i+1}.jpg"
                aug_path = output_class_dir / aug_filename
                cv2.imwrite(str(aug_path), aug_img)
                total_augmented += 1
    
    print(f"\nGenerated {total_augmented} augmented images from {total_original} originals")
    print(f"   Ratio: {n_augmented_per_sample} augmented per original")
