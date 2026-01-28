#!/usr/bin/env python3
"""
Main script for running Polish Sign Language Recognition experiments.
"""

import os
import sys
import warnings
import argparse
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['GLOG_logtostderr'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
import absl.logging

logging.getLogger().setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)

warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore",
    message="y_pred contains classes not in y_true",
    category=UserWarning,
    module="sklearn.metrics",
)

from src.config import Config
from src.data_loader import load_raw_data_and_extract_features, load_augmented_features
from src.cross_validation import run_sklearn_cv, run_keras_mlp_cv, run_cnn_cv
from src.report import write_report


def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(title)
    print("="*70)


def main():
    """Main experiment runner with Stratified K-Fold CV."""
    parser = argparse.ArgumentParser(description="Run PSL recognition experiments")
    parser.add_argument(
        "--with-augmentation",
        action="store_true",
        help="Run both experiments (with and without augmentation) and compare"
    )
    args = parser.parse_args()
    
    print_header("POLISH SIGN LANGUAGE RECOGNITION - CROSS-VALIDATION EXPERIMENTS")
    print("\nMetoda: Stratified K-Fold Cross-Validation")
    
    script_dir = Path(__file__).parent.resolve()
    paths = Config.get_paths(script_dir)
    paths['results_dir'].mkdir(parents=True, exist_ok=True)
    
    if args.with_augmentation:
        print("Mode: COMPARISON (with and without augmentation)")
        print("This will run experiments twice and generate comparison report")
    
    print(f"\nConfiguration:")
    print(f"  K-folds: {Config.N_SPLITS}")
    print(f"  Random seed: {Config.SEED}")
    print(f"  Raw data: {paths['raw_dir']}")
    print(f"  Results: {paths['results_dir']}")
    
    all_results = []
    
    if args.with_augmentation:
        print_header("PART 1: EXPERIMENTS WITH AUGMENTATION")
        augmented_dir = script_dir / "data" / "augmented"
        if not augmented_dir.exists() or not list(augmented_dir.glob("*/*.jpg")):
            print(f"\nError: Augmented data not found!")
            print(f"   Please run: python3 generate_augmented_data.py")
            return
        
        print("\nLoading augmented data (this may take a while)...")
        augmented_data = load_augmented_features(
            raw_dir=paths['raw_dir'],
            augmented_dir=augmented_dir
        )
        
        X_aug = augmented_data['X'][~augmented_data['is_augmented']]
        y_aug = augmented_data['y'][~augmented_data['is_augmented']]
        
        results_augmented = run_all_models(
            X_aug, y_aug, paths, 
            use_augmentation=True,
            augmented_data=augmented_data
        )
        all_results.extend(results_augmented)
        
        print_header("PART 2: EXPERIMENTS WITHOUT AUGMENTATION (BASELINE)")
        X, y = load_raw_data_and_extract_features(paths['raw_dir'])
        results_baseline = run_all_models(X, y, paths, use_augmentation=False)
        all_results.extend(results_baseline)
        
        print_header("GENERATING COMPARISON REPORT")
        report_path = paths['results_dir'] / "report_cv_comparison.md"
        write_report(
            all_results,
            report_path,
            n_splits=Config.N_SPLITS,
            total_samples=len(X),
            comparison_mode=True
        )
    else:
        print_header("LOADING DATA")
        X, y = load_raw_data_and_extract_features(paths['raw_dir'])
        
        print(f"\nLoaded:")
        print(f"  X.shape: {X.shape}")
        print(f"  y.shape: {y.shape}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Classes: {len(set(y))}")
        
        all_results = run_all_models(X, y, paths, use_augmentation=False)
        
        print_header("GENERATING REPORT")
        report_path = paths['results_dir'] / "report_cv.md"
        write_report(
            all_results,
            report_path,
            n_splits=Config.N_SPLITS,
            total_samples=len(X),
            comparison_mode=False
        )
    
    print_header("SUMMARY - COMPARISON OF ALL MODELS")
    print(f"\n{'Model':<30} {'Accuracy':<25} {'F1-macro':<25}")
    print("-"*80)
    
    for result in all_results:
        print(
            f"{result['model']:<30} "
            f"{result['acc_mean']:.4f} ± {result['acc_std']:.4f}        "
            f"{result['f1_macro_mean']:.4f} ± {result['f1_macro_std']:.4f}"
        )
    
    print_header("ALL EXPERIMENTS COMPLETED!")
    print(f"Results saved in: {paths['results_dir']}")
    print("="*70)


def run_all_models(X, y, paths, use_augmentation=False, augmented_data=None):
    """Run all models and return results."""
    results = []
    
    suffix = " (Augmented)" if use_augmentation else ""
    
    sklearn_models = [
        (
            RandomForestClassifier(
                random_state=Config.SEED, 
                n_estimators=Config.RF_N_ESTIMATORS
            ),
            f"RandomForest{suffix}"
        ),
        (
            LogisticRegression(
                max_iter=Config.LR_MAX_ITER, 
                random_state=Config.SEED
            ),
            f"LogisticRegression{suffix}"
        ),
    ]
    
    for model, model_name in sklearn_models:
        result = run_sklearn_cv(
            X, y,
            model=model,
            model_name=model_name,
            n_splits=Config.N_SPLITS,
            seed=Config.SEED,
            results_dir=paths['results_dir'],
            augmented_data=augmented_data,
            use_augmentation=use_augmentation
        )
        results.append(result)
    
    keras_result = run_keras_mlp_cv(
        X, y,
        n_splits=Config.N_SPLITS,
        seed=Config.SEED,
        results_dir=paths['results_dir'],
        augmented_data=augmented_data,
        use_augmentation=use_augmentation
    )
    keras_result['model'] = f"Keras MLP{suffix}"
    results.append(keras_result)
    
    # CNN - pass augmented images directory if using augmentation
    augmented_img_dir = None
    if use_augmentation and augmented_data:
        script_dir = Path(__file__).parent.resolve()
        augmented_img_dir = script_dir / "data" / "augmented" / "images"
    
    cnn_result = run_cnn_cv(
        raw_dir=paths['raw_dir'],
        n_splits=Config.N_SPLITS,
        seed=Config.SEED,
        img_size=Config.IMG_SIZE,
        results_dir=paths['results_dir'],
        augmented_img_dir=augmented_img_dir,
        use_augmentation=use_augmentation
    )
    cnn_result['model'] = f"CNN{suffix}"
    results.append(cnn_result)
    
    return results


if __name__ == "__main__":
    main()
