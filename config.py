# train_images_path = '/content/train_images'
# train_masks_path = '/content/train_masks'
# test_images_path = '/content/test_images'
# test_masks_path = '/content/test_masks'
# train_metadata_path = '/content/train-metadata.csv'
# test_metadata_path = '/content/test-metadata.csv'

# batch_size = 8
# num_epochs = 50
# num_classes = 1
# learning_rate = 1e-3



import os

# Base directory for the data
BASE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'flood-data')

# Define the paths to images, masks, and metadata
images_path = os.path.join(BASE_DATA_DIR, 'Image')  # Assuming all images are in the 'Image' folder
masks_path = os.path.join(BASE_DATA_DIR, 'Mask')    # Assuming all masks are in the 'Mask' folder
metadata_path = os.path.join(BASE_DATA_DIR, 'metadata.csv')  # Path to metadata file

# Hyperparameters and other configuration settings
batch_size = 8
num_epochs = 50
num_classes = 1
learning_rate = 1e-3
