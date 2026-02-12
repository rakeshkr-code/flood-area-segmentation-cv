import torch
import numpy as np
from PIL import Image
import os
import logging
from torchvision import transforms

# ## INFERENCE CLASS ======================================================================>
class Predictor:
    """Handles model inference"""
    
    def __init__(self, model, config, visualizer=None):
        self.model = model
        self.config = config
        self.visualizer = visualizer
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(config.test_img_size),
            transforms.ToTensor()
        ])
        
        logging.info(f"Predictor initialized on device: {self.device}")
    
    def load_model(self, checkpoint_path):
        """Load model weights"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            logging.info(f"Loaded model from {checkpoint_path}")
        else:
            logging.error(f"Checkpoint {checkpoint_path} not found!")
    
    def predict_single(self, image_path, visualize=True, save_path=None, alpha=None):
        """Predict on a single image"""
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)
        
        # Transform
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(image_tensor)
            pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # Threshold
        pred_mask = (pred_mask > 0.5).astype(np.float32)
        
        # Visualize if requested
        if visualize and self.visualizer:
            alpha = alpha if alpha is not None else self.config.mask_alpha
            self.visualizer.plot_image_with_mask(
                original_image, pred_mask, alpha=alpha,
                title="Prediction", save_path=save_path
            )
        
        return pred_mask
    
    def predict_batch(self, dataloader, num_samples=5, save_dir=None):
        """Predict on a batch of images"""
        save_dir = save_dir or self.config.predictions_dir
        
        batch = next(iter(dataloader))
        images = batch['X'][:num_samples].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(images)
            pred_masks = torch.sigmoid(outputs).cpu().numpy()
        
        # Threshold
        pred_masks = (pred_masks > 0.5).astype(np.float32)
        
        # Visualize
        if self.visualizer:
            for i in range(num_samples):
                img = images[i].cpu()
                mask = pred_masks[i][0]
                
                save_path = os.path.join(save_dir, f"prediction_{i+1}.png")
                self.visualizer.plot_image_with_mask(
                    img, mask, alpha=self.config.mask_alpha,
                    title=f"Prediction {i+1}", save_path=save_path
                )
        
        return pred_masks
