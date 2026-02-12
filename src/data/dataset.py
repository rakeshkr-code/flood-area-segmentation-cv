import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import logging

# ## DATASET CLASS =============================================================>
class SyncedRandomTransform:
    """Synchronized transformations for image and mask"""
    
    def __init__(self, img_size=(576, 576), for_training=False):
        self.img_size = img_size
        self.for_training = for_training
        
        self.transform_image = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])
        
        self.transform_mask = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])
        
        if self.for_training:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            )
    
    def __call__(self, image, mask):
        if self.for_training:
            # Color jitter (image only)
            if random.random() < 0.4:
                image = self.color_jitter(image)
            
            # Horizontal flip
            if random.random() > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            
            # Vertical flip
            if random.random() > 0.5:
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)
            
            # Rotation
            if random.random() < 0.4:
                angle = random.uniform(-10, 10)
                image = transforms.functional.rotate(image, angle)
                mask = transforms.functional.rotate(mask, angle)
        
        image = self.transform_image(image)
        mask = self.transform_mask(mask)
        
        return image, mask


class FloodDataset(Dataset):
    """Dataset class for flood segmentation"""
    
    def __init__(self, metadata_file, images_path, masks_path, transform=None):
        self.metadata = pd.read_csv(metadata_file)
        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        
        logging.info(f"Loaded dataset with {len(self.metadata)} samples from {metadata_file}")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.images_path, self.metadata.iloc[idx]['Image'])
        mask_name = os.path.join(self.masks_path, self.metadata.iloc[idx]['Mask'])
        
        image = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')
        
        if self.transform:
            image, mask = self.transform(image, mask)
        
        mask = torch.squeeze(mask, 0)
        
        return image, mask
    
    def get_sample_names(self, idx):
        """Get filenames for a specific index"""
        return (self.metadata.iloc[idx]['Image'],
                self.metadata.iloc[idx]['Mask'])
