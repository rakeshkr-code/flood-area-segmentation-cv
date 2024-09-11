import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SyncdRandomTransform:
    """
    Synchronized Random Transformation
    A custom transformation class to apply consistent/synchronized transformations to both images and masks.
    """
    def __init__(self, for_training=False):
        self.for_training = for_training
        
        if self.for_training:
            # Training Transformation for Image
            self.transform_image = transforms.Compose([
                transforms.Resize((576, 576)),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Imagenet normalization
            ])
            # Training Transformation for Mask
            self.transform_mask = transforms.Compose([
                transforms.Resize((576, 576)),
                transforms.ToTensor()
            ])
            # Define transformations for ColorJitter
            self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        else:
            # Testing Transformation for Image
            self.transform_image = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ])
            # Testing Transformation for Mask
            self.transform_mask = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor()
            ])

    def __call__(self, image, mask):
        # Flips and Rotations will be imlemented during the training only
        if self.for_training:
            # Apply ColorJitter 40% of the time (will be applied only on image)
            if random.random() < 0.4:
                image = self.color_jitter(image)

            # Apply random horizontal flip for 50% of the samples
            if random.random() > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            
            # Apply random vertical flip for 50% of the samples
            if random.random() > 0.5:
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)
            
            # Apply random rotation for 40% of the samples
            if random.random() < 0.4:
                angle = random.uniform(-10, 10)
                image = transforms.functional.rotate(image, angle)
                mask = transforms.functional.rotate(mask, angle)

        # Apply transformations
        image = self.transform_image(image)
        mask = self.transform_mask(mask)

        return image, mask

class FloodDataset(Dataset):
    def __init__(self, metadata_file, base_image_folder, base_mask_folder, transform=None):
        """
        Custom dataset for loading flood images and masks.

        Args:
            metadata_file (str): Path to the metadata CSV file.
            base_image_folder (str): Base path to the images directory.
            base_mask_folder (str): Base path to the masks directory.
            transform (callable, optional): A function/transform to apply to the images and masks.
        """
        try:
            self.metadata = pd.read_csv(metadata_file)
            self.base_image_folder = base_image_folder
            self.base_mask_folder = base_mask_folder
            self.transform = transform
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            raise

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        try:
            img_name = os.path.join(self.base_image_folder, self.metadata.iloc[idx]['Image'])
            mask_name = os.path.join(self.base_mask_folder, self.metadata.iloc[idx]['Mask'])

            image = Image.open(img_name).convert('RGB')
            mask = Image.open(mask_name).convert('L')

            if self.transform:
                # image = self.transform(image)
                # mask = self.transform(mask)
                image, mask = self.transform(image, mask)  # Apply custom transform
            
            mask = torch.squeeze(mask, 0)  # Remove extra dimension

            return image, mask
        except Exception as e:
            logging.error(f"Error processing image or mask at index {idx}: {e}")
            raise

def create_dataloaders(metadata_path, images_path, masks_path, batch_size, split_ratio=0.8):
    """
    Create data loaders for the training and testing datasets.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        images_path (str): Base path to the images directory.
        masks_path (str): Base path to the masks directory.
        batch_size (int): Batch size for data loading.
        split_ratio (float): Ratio to split the data into training and testing datasets.

    Returns:
        tuple: Train and test DataLoader objects.
    """
    # train_transform = transforms.Compose([
    #     transforms.Resize((576, 576)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     # transforms.RandomRotation(10),
    #     # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    #     transforms.ToTensor(),
    #     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Imagenet normalization
    # ])
    # test_transform = transforms.Compose([
    #     transforms.Resize((512, 512)),
    #     transforms.ToTensor(),
    # ])
    train_transform = SyncdRandomTransform(for_training=True)
    test_transform = SyncdRandomTransform()

    # Load metadata
    metadata = pd.read_csv(metadata_path)

    # Split metadata into train and test based on split_ratio
    train_metadata = metadata.sample(frac=split_ratio, random_state=42).reset_index(drop=True)
    test_metadata = metadata.drop(train_metadata.index).reset_index(drop=True)

    # Save split metadata to temporary CSV files
    train_metadata.to_csv('train_metadata_temp.csv', index=False)
    test_metadata.to_csv('test_metadata_temp.csv', index=False)

    # Create datasets
    train_dataset = FloodDataset('train_metadata_temp.csv', images_path, masks_path, train_transform)
    test_dataset = FloodDataset('test_metadata_temp.csv', images_path, masks_path, test_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True)

    return train_loader, test_loader

def custom_collate_fn(batch):
    """
    Custom collate function to handle batches of images and masks.

    Args:
        batch (list): A list of tuples where each tuple contains an image and its corresponding mask.

    Returns:
        dict: A dictionary containing batched images and masks.
    """
    images, masks = zip(*batch)
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    return {'X': images, 'y': masks}
