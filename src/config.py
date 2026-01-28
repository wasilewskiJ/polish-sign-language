"""
Configuration module for PSL experiments.
"""

from pathlib import Path


class Config:
    """Central configuration for experiments."""
    
    SEED = 666
    N_SPLITS = 5
    IMG_SIZE = (128, 128)
    
    KERAS_EPOCHS = 50
    KERAS_BATCH_SIZE = 32
    KERAS_LEARNING_RATE = 0.001
    KERAS_L2_REG = 0.005
    KERAS_DROPOUT = 0.3
    
    CNN_EPOCHS = 20
    CNN_BATCH_SIZE = 32
    CNN_LEARNING_RATE = 0.001
    CNN_EARLY_STOPPING_PATIENCE = 5
    
    RF_N_ESTIMATORS = 100
    LR_MAX_ITER = 1000
    
    LANDMARK_FEATURES = 78  # 63 raw coords + 15 relationships
    
    @staticmethod
    def get_paths(script_dir: Path):
        """Get all relevant paths based on script directory."""
        paths = {
            'raw_dir': script_dir / "data" / "raw",
            'augmented_dir': script_dir / "data" / "augmented",
            'results_dir': script_dir / "results_cv",
        }
        return paths


def get_paths():

    return Config.get_paths(Path.cwd())
