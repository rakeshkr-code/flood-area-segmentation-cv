import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import logging
from .dataset import FloodDataset, SyncedRandomTransform

# ## DATALOADER MANAGER =====================================================>
class DataLoaderManager:
    """Manages data loading and splitting"""
    
    def __init__(self, config):
        self.config = config
        self.train_loader = None
        self.test_loader = None
        self.visualizer = None
    
    def setup_dataloaders(self, visualize=False):
        """Create train and test dataloaders"""
        # Split metadata
        self._split_metadata()
        
        # Create transforms
        train_transform = SyncedRandomTransform(
            img_size=self.config.train_img_size,
            for_training=True
        )
        test_transform = SyncedRandomTransform(
            img_size=self.config.test_img_size,
            for_training=False
        )
        
        # Create datasets
        train_dataset = FloodDataset(
            self.config.train_metadata_path,
            self.config.images_path,
            self.config.masks_path,
            train_transform
        )
        
        test_dataset = FloodDataset(
            self.config.test_metadata_path,
            self.config.images_path,
            self.config.masks_path,
            test_transform
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self._custom_collate_fn,
            pin_memory=True if self.config.device == 'cuda' else False
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self._custom_collate_fn,
            pin_memory=True if self.config.device == 'cuda' else False
        )
        
        logging.info(f"Created dataloaders - Train: {len(self.train_loader.dataset)}, "
                    f"Test: {len(self.test_loader.dataset)}")
        
        # Visualize if requested
        if visualize and self.visualizer:
            self.visualizer.visualize_from_dataloader(
                self.train_loader, n=4,
                save_path=f"{self.config.plots_dir}/dataloader_samples.png"
            )
        
        return self.train_loader, self.test_loader
    
    def _split_metadata(self):
        """Split metadata into train and test sets"""
        metadata = pd.read_csv(self.config.metadata_path)
        
        # Shuffle
        metadata = metadata.sample(frac=1, random_state=self.config.random_seed).reset_index(drop=True)
        
        # Split
        split_idx = int(len(metadata) * self.config.train_split_ratio)
        train_metadata = metadata[:split_idx]
        test_metadata = metadata[split_idx:]
        
        # Save
        train_metadata.to_csv(self.config.train_metadata_path, index=False)
        test_metadata.to_csv(self.config.test_metadata_path, index=False)
        
        logging.info(f"Split metadata - Train: {len(train_metadata)}, Test: {len(test_metadata)}")
    
    @staticmethod
    def _custom_collate_fn(batch):
        """Custom collate function"""
        images, masks = zip(*batch)
        images = torch.stack(images, dim=0)
        masks = torch.stack(masks, dim=0)
        return {'X': images, 'y': masks}
    
    def attach_visualizer(self, visualizer):
        """Attach visualizer for dataloader visualization"""
        self.visualizer = visualizer

if __name__ == "__main__":
    """Test dataloader creation and visualization"""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    
    from src.config import Config
    from src.utils.visualization import Visualizer
    
    config = Config()
    viz = Visualizer(config)
    
    # Test dataloader creation
    print("Creating dataloaders...")
    data_manager = DataLoaderManager(config)
    data_manager.attach_visualizer(viz)
    train_loader, test_loader = data_manager.setup_dataloaders(visualize=True)
    
    print(f"✓ Train samples: {len(train_loader.dataset)}")
    print(f"✓ Test samples: {len(test_loader.dataset)}")
    
    # Test batch retrieval
    batch = next(iter(train_loader))
    print(f"✓ Batch shape - Images: {batch['X'].shape}, Masks: {batch['y'].shape}")