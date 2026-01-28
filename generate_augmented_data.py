#!/usr/bin/env python3
"""
Script to pre-generate augmented images for all samples.
Run this once before experiments to create augmented versions.

Flow:
1. For each image in data/raw/<LETTER>/<LETTER><ID>.jpg
2. Generate N augmented versions
3. Save to data/augmented/<LETTER>/<LETTER><ID>_aug<N>.jpg

Example: K21.jpg -> K21_aug0.jpg, K21_aug1.jpg, ..., K21_aug4.jpg
"""

import os
import cv2
from pathlib import Path
from tqdm import tqdm

from src.augmentation import augment_image
from src.config import get_paths


NUM_AUGMENTATIONS = 5


def generate_augmented_images(raw_dir, output_dir, n_augmented=NUM_AUGMENTATIONS):
    """Generate augmented images for all raw images."""
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    
    total_original = 0
    total_augmented = 0
    
    for class_dir in sorted(raw_path.iterdir()):
        if not class_dir.is_dir():
            continue
            
        letter = class_dir.name
        output_class_dir = output_path / letter
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = sorted(class_dir.glob('*.jpg'))
        print(f"\nProcessing {letter}... ({len(image_files)} images)")
        
        for img_path in tqdm(image_files, desc=f"  {letter}", leave=False):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            total_original += 1
            
            # Generate N augmented versions
            for aug_idx in range(n_augmented):
                augmented = augment_image(img)
                
                # Save with naming: <LETTER><ID>_aug<N>.jpg
                # Example: K21.jpg -> K21_aug0.jpg, K21_aug1.jpg, etc.
                base_name = img_path.stem  # e.g., "K21"
                aug_name = f"{base_name}_aug{aug_idx}.jpg"
                output_file = output_class_dir / aug_name
                
                cv2.imwrite(str(output_file), augmented)
                total_augmented += 1
    
    return total_original, total_augmented


def main():
    paths = get_paths()
    
    print("="*70)
    print("AUGMENTED IMAGE GENERATION")
    print("="*70)
    print(f"Raw images:        {paths['raw_dir']}")
    print(f"Output directory:  {paths['augmented_dir']}")
    print(f"Augmentations/img: {NUM_AUGMENTATIONS}")
    print("="*70)
    
    os.makedirs(paths['augmented_dir'], exist_ok=True)
    
    n_original, n_augmented = generate_augmented_images(
        raw_dir=paths['raw_dir'],
        output_dir=paths['augmented_dir'],
        n_augmented=NUM_AUGMENTATIONS
    )
    
    print("\n" + "="*70)
    print("Augmented image generation complete!")
    print(f"   Original images:   {n_original}")
    print(f"   Augmented images:  {n_augmented}")
    print(f"   Saved to:          {paths['augmented_dir']}")
    print("="*70)


if __name__ == '__main__':
    main()
