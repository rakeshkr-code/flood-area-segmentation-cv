# import os
# from dataclasses import dataclass, field
# from typing import Tuple
# from pathlib import Path

# PROJECT_ROOT = Path(__file__).resolve().parent.parent

# # ## CONFIGURATION MANAGER ===============================================================>
# @dataclass
# class Config:
#     """Configuration class for flood segmentation project"""
    
#     # Paths
#     base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     data_dir: str = field(default_factory=lambda: os.path.join(Config.base_dir, 'archive'))
#     images_path: str = field(default_factory=lambda: os.path.join(Config.data_dir, 'Image'))
#     masks_path: str = field(default_factory=lambda: os.path.join(Config.data_dir, 'Mask'))
#     metadata_path: str = field(default_factory=lambda: os.path.join(Config.data_dir, 'metadata.csv'))
    
#     # raw_movies_path: Path = PROJECT_ROOT / "data" / "raw" / "tmdb_5000_movies.csv"
#     train_metadata_path: str = 'data/train_metadata.csv'
#     test_metadata_path: str = 'data/test_metadata.csv'
    
#     checkpoint_dir: str = 'checkpoints'
#     log_dir: str = 'logs'
#     output_dir: str = 'outputs'
#     plots_dir: str = 'outputs/plots'
#     predictions_dir: str = 'outputs/predictions'
    
#     # Training parameters
#     batch_size: int = 8
#     num_epochs: int = 50
#     num_classes: int = 1
#     learning_rate: float = 1e-3
#     weight_decay: float = 1e-5
    
#     # Scheduler parameters
#     scheduler_step_size: int = 8
#     scheduler_gamma: float = 0.1
    
#     # Data split
#     train_split_ratio: float = 0.8
#     random_seed: int = 42
    
#     # Image parameters
#     train_img_size: Tuple[int, int] = (576, 576)
#     test_img_size: Tuple[int, int] = (512, 512)
    
#     # Training settings
#     save_interval: int = 10  # Save plots every N epochs
#     resume_training: bool = False
#     resume_checkpoint: str = None
    
#     # Loss weights
#     bce_weight: float = 0.5
#     dice_weight: float = 0.5
    
#     # Visualization
#     mask_color: Tuple[int, int, int] = (255, 0, 0)  # Red
#     mask_alpha: float = 0.4  # Transparency
    
#     # Device
#     device: str = 'cuda'
#     num_workers: int = 4
    
#     def __post_init__(self):
#         """Create necessary directories"""
#         os.makedirs(self.checkpoint_dir, exist_ok=True)
#         os.makedirs(self.log_dir, exist_ok=True)
#         os.makedirs(self.output_dir, exist_ok=True)
#         os.makedirs(self.plots_dir, exist_ok=True)
#         os.makedirs(self.predictions_dir, exist_ok=True)
#         os.makedirs(os.path.dirname(self.train_metadata_path), exist_ok=True)

# -----------------------------------------------------------------------------------

from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class Config:
    """Configuration class for flood segmentation project"""
    
    # Project root - resolve from this file's location
    PROJECT_ROOT: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    
    # Paths - all relative to PROJECT_ROOT
    data_dir: Path = field(init=False)
    images_path: Path = field(init=False)
    masks_path: Path = field(init=False)
    metadata_path: Path = field(init=False)
    train_metadata_path: Path = field(init=False)
    test_metadata_path: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    log_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    plots_dir: Path = field(init=False)
    predictions_dir: Path = field(init=False)
    
    # Training parameters
    batch_size: int = 8
    num_epochs: int = 10
    num_classes: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # Scheduler parameters
    scheduler_step_size: int = 8
    scheduler_gamma: float = 0.1
    
    # Data split
    train_split_ratio: float = 0.8
    random_seed: int = 42
    
    # Image parameters
    train_img_size: Tuple[int, int] = (576, 576)
    test_img_size: Tuple[int, int] = (512, 512)
    
    # Training settings
    save_interval: int = 10
    resume_training: bool = False
    resume_checkpoint: str = None
    
    # Loss weights
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    
    # Visualization
    mask_color: Tuple[int, int, int] = (255, 0, 0)
    mask_alpha: float = 0.4
    
    # Device
    device: str = 'cuda'
    num_workers: int = 4
    
    def __post_init__(self):
        """Initialize paths and create directories"""
        # Set all paths relative to PROJECT_ROOT
        self.data_dir = self.PROJECT_ROOT / 'archive'
        self.images_path = self.data_dir / 'Image'
        self.masks_path = self.data_dir / 'Mask'
        self.metadata_path = self.data_dir / 'metadata.csv'
        
        self.train_metadata_path = self.PROJECT_ROOT / 'data' / 'train_metadata.csv'
        self.test_metadata_path = self.PROJECT_ROOT / 'data' / 'test_metadata.csv'
        
        self.checkpoint_dir = self.PROJECT_ROOT / 'checkpoints'
        self.log_dir = self.PROJECT_ROOT / 'logs'
        self.output_dir = self.PROJECT_ROOT / 'outputs'
        self.plots_dir = self.output_dir / 'plots'
        self.predictions_dir = self.output_dir / 'predictions'
        
        # Create necessary directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.train_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    def get_path(self, path_type: str) -> Path:
        """Utility to get any path by name"""
        return getattr(self, f"{path_type}_path", None)


# -----------------------------------------------------------------------------------

# # if __name__ == "__main__":
# #     print(f"PROJECT_ROOT: {PROJECT_ROOT}")
# #     config = Config()
# #     print("Configuration initialized:")
# #     print(config)
