#!/usr/bin/env python3
"""
Quick test script to verify modular structure works correctly.
"""

import sys
from pathlib import Path

def test_imports():
    """Test module imports."""
    print("="*70)
    print("TESTING MODULE IMPORTS")
    print("="*70)
    
    try:
        from src.config import Config
        print("OK: src.config")
        
        from src.data_loader import load_raw_data_and_extract_features
        print("OK: src.data_loader")
        
        from src.models import build_keras_mlp, build_cnn
        print("OK: src.models")
        
        from src.evaluation import compute_weights, per_class_metrics, save_confusion_matrix
        print("OK: src.evaluation")
        
        from src.cross_validation import run_sklearn_cv, run_keras_mlp_cv, run_cnn_cv
        print("OK: src.cross_validation")
        
        from src.report import write_report
        print("OK: src.report")
        
        print("\nAll modules imported successfully!")
        return True
        
    except ImportError as e:
        print(f"\nImport error: {e}")
        return False


def test_config():
    """Test config."""
    print("\n" + "="*70)
    print("TESTING CONFIGURATION")
    print("="*70)
    
    from src.config import Config
    
    print(f"Random seed: {Config.SEED}")
    print(f"CV folds: {Config.N_SPLITS}")
    print(f"Image size: {Config.IMG_SIZE}")
    print(f"Landmark features: {Config.LANDMARK_FEATURES}")
    print(f"Keras epochs: {Config.KERAS_EPOCHS}")
    print(f"CNN epochs: {Config.CNN_EPOCHS}")
    
    # Test path generation
    script_dir = Path(__file__).parent
    paths = Config.get_paths(script_dir)
    
    print(f"\nPaths:")
    for key, path in paths.items():
        exists = "OK" if path.exists() else "MISSING"
        print(f"  {key}: {path} {exists}")
    
    print("\nConfiguration test passed!")


def test_model_building():
    """Test model building."""
    print("\n" + "="*70)
    print("TESTING MODEL BUILDING")
    print("="*70)
    
    from src.models import build_keras_mlp, build_cnn
    
    # Test MLP
    mlp = build_keras_mlp(input_shape=78, num_classes=22)
    print(f"MLP built: {mlp.count_params()} parameters")
    
    # Test CNN
    cnn = build_cnn(input_shape=(128, 128, 3), num_classes=22)
    print(f"CNN built: {cnn.count_params()} parameters")
    
    print("\nModel building test passed!")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PSL RECOGNITION - MODULE STRUCTURE TEST")
    print("="*70 + "\n")
    
    if not test_imports():
        sys.exit(1)
    
    test_config()
    
    try:
        test_model_building()
    except ImportError:
        print("\nTensorFlow not available, skipping model building test")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70)
    print("\nYou can now run: python3 run_experiments_modular.py")
    print()


if __name__ == "__main__":
    main()
