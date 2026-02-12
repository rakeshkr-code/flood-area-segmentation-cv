import torch
import logging
from src.config import Config
from src.data.dataloader import DataLoaderManager
from src.models.unet import ResNetUNet
from src.training.trainer import Trainer
from src.utils.visualization import Visualizer

# ## MAIN TRAINING SCRIPT ================================================================>
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)

def main():
    # Load config
    config = Config()
    
    # Create visualizer
    visualizer = Visualizer(config)
    
    # Setup data
    data_manager = DataLoaderManager(config)
    data_manager.attach_visualizer(visualizer)
    train_loader, val_loader = data_manager.setup_dataloaders(visualize=True)
    
    # Create model
    model = ResNetUNet(config.num_classes)
    
    # Create trainer
    trainer = Trainer(model, config, train_loader, val_loader)
    
    # Train
    trained_model, train_losses, val_losses = trainer.train()
    
    logging.info("Training completed successfully!")

if __name__ == "__main__":
    main()
