import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from typing import Union, List, Optional
import os

# ## VISUALIZATION MODULE ==============================================================>
class Visualizer:
    """Handles all visualization tasks for images and masks"""
    
    def __init__(self, config):
        self.config = config
        self.mask_color = np.array(config.mask_color) / 255.0
        self.mask_alpha = config.mask_alpha
    
    def plot_image(self, image: Union[str, np.ndarray, torch.Tensor], 
                   title: str = "Image", save_path: Optional[str] = None):
        """Plot a single image"""
        img = self._load_image(image)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.show()
    
    def plot_mask(self, mask: Union[str, np.ndarray, torch.Tensor],
                  title: str = "Mask", save_path: Optional[str] = None):
        """Plot a single mask"""
        msk = self._load_mask(mask)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(msk, cmap='gray')
        plt.title(title)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.show()
    
    def plot_image_with_mask(self, image: Union[str, np.ndarray, torch.Tensor],
                            mask: Union[str, np.ndarray, torch.Tensor],
                            alpha: Optional[float] = None,
                            title: str = "Image with Mask Overlay",
                            save_path: Optional[str] = None):
        """Plot image with mask overlay in light red with transparency"""
        img = self._load_image(image)
        msk = self._load_mask(mask)
        
        alpha = alpha if alpha is not None else self.mask_alpha
        
        # Create overlay
        overlay = self._create_overlay(img, msk, alpha)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(overlay)
        plt.title(title)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.show()
    
    def plot_comparison(self, image: Union[str, np.ndarray, torch.Tensor],
                       mask: Union[str, np.ndarray, torch.Tensor],
                       alpha: Optional[float] = None,
                       save_path: Optional[str] = None):
        """Plot image, mask, and overlay side by side"""
        img = self._load_image(image)
        msk = self._load_mask(mask)
        alpha = alpha if alpha is not None else self.mask_alpha
        overlay = self._create_overlay(img, msk, alpha)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(msk, cmap='gray')
        axes[1].set_title('Mask')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.show()
    
    def plot_batch(self, images: List, masks: List, n: int = 4,
                   alpha: Optional[float] = None, save_path: Optional[str] = None):
        """Plot first n images with mask overlays"""
        n = min(n, len(images))
        alpha = alpha if alpha is not None else self.mask_alpha
        
        fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
        if n == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(n):
            img = self._load_image(images[i])
            msk = self._load_mask(masks[i])
            overlay = self._create_overlay(img, msk, alpha)
            
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'Image {i+1}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(overlay)
            axes[1, i].set_title(f'With Mask Overlay')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.show()
    
    def visualize_from_dataloader(self, dataloader, n: int = 4,
                                  alpha: Optional[float] = None,
                                  save_path: Optional[str] = None):
        """Visualize samples from dataloader"""
        batch = next(iter(dataloader))
        images = batch['X'][:n]
        masks = batch['y'][:n]
        
        alpha = alpha if alpha is not None else self.mask_alpha
        
        fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
        if n == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(n):
            img = self._tensor_to_image(images[i])
            msk = self._tensor_to_mask(masks[i])
            overlay = self._create_overlay(img, msk, alpha)
            
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'Image {i+1}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(overlay)
            axes[1, i].set_title(f'With Mask Overlay')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.show()
    
    def _load_image(self, image: Union[str, np.ndarray, torch.Tensor]) -> np.ndarray:
        """Load image from various formats"""
        if isinstance(image, str):
            # Load from file path
            if not os.path.isabs(image):
                image = os.path.join(self.config.images_path, image)
            img = np.array(Image.open(image).convert('RGB'))
        elif isinstance(image, torch.Tensor):
            img = self._tensor_to_image(image)
        else:
            img = np.array(image)
        
        # Normalize to [0, 1] if needed
        if img.max() > 1.0:
            img = img / 255.0
        
        return img
    
    def _load_mask(self, mask: Union[str, np.ndarray, torch.Tensor]) -> np.ndarray:
        """Load mask from various formats"""
        if isinstance(mask, str):
            # Load from file path
            if not os.path.isabs(mask):
                mask = os.path.join(self.config.masks_path, mask)
            msk = np.array(Image.open(mask).convert('L'))
        elif isinstance(mask, torch.Tensor):
            msk = self._tensor_to_mask(mask)
        else:
            msk = np.array(mask)
        
        # Normalize to [0, 1]
        if msk.max() > 1.0:
            msk = msk / 255.0
        
        return msk
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy image"""
        if tensor.dim() == 4:
            tensor = tensor[0]
        img = tensor.permute(1, 2, 0).cpu().numpy()
        return np.clip(img, 0, 1)
    
    def _tensor_to_mask(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy mask"""
        if tensor.dim() == 3:
            tensor = tensor[0]
        msk = tensor.cpu().numpy()
        return np.clip(msk, 0, 1)
    
    def _create_overlay(self, image: np.ndarray, mask: np.ndarray,
                       alpha: float) -> np.ndarray:
        """Create overlay of mask on image"""
        overlay = image.copy()
        
        # Ensure mask is binary
        mask_binary = (mask > 0.5).astype(float)
        
        # Create red mask
        red_mask = np.zeros_like(overlay)
        red_mask[:, :, 0] = mask_binary  # Red channel
        
        # Blend
        overlay = (1 - alpha * mask_binary[:, :, np.newaxis]) * overlay + \
                  alpha * red_mask
        
        return np.clip(overlay, 0, 1)


if __name__ == "__main__":
    """Test visualization module independently"""
    from src.config import Config
    import pandas as pd
    
    config = Config()
    viz = Visualizer(config)
    
    # Test 1: Load and plot single image
    print("Testing single image visualization...")
    metadata = pd.read_csv(config.metadata_path)
    sample_img = metadata.iloc[0]['Image']
    sample_mask = metadata.iloc[0]['Mask']
    
    viz.plot_comparison(sample_img, sample_mask, 
                       save_path='test_visualization.png')
    print("✓ Visualization test passed")
