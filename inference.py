import logging
from src.config import Config
from src.models.unet import ResNetUNet
from src.inference.predictor import Predictor
from src.utils.visualization import Visualizer
from src.data.dataloader import DataLoaderManager

# ## INFERENCE SCRIPT ===============================================================>

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Load config
    config = Config()
    
    # Create visualizer
    visualizer = Visualizer(config)
    
    # Create model
    model = ResNetUNet(config.num_classes)
    
    # Create predictor
    predictor = Predictor(model, config, visualizer)
    predictor.load_model('checkpoints/best_model.pth')
    
    # Option 1: Predict on single image
    predictor.predict_single(
        'test_flood_img.jpg',
        visualize=True,
        save_path='outputs/predictions/single_prediction.png',
        alpha=0.4
    )
    
    # Option 2: Predict on batch from dataloader
    data_manager = DataLoaderManager(config)
    _, test_loader = data_manager.setup_dataloaders()
    predictor.predict_batch(test_loader, num_samples=5)

if __name__ == "__main__":
    main()
