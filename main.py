import torch
from data_loading import create_dataloaders
from model import ResNetUNet
from train import train_model
import logging
from config import *
from utils import plot_loss_curves

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Device: {device}')

    model = ResNetUNet(num_classes).to(device)
    # train_loader, test_loader = create_dataloaders(train_images_path, train_masks_path, train_metadata_path, test_images_path, test_masks_path, test_metadata_path, batch_size)
    train_loader, test_loader = create_dataloaders(metadata_path, images_path, masks_path, batch_size, split_ratio=0.8)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    trained_model, train_losses, val_losses = train_model(model, optimizer, scheduler, train_loader, test_loader, device, num_epochs=num_epochs)

    plot_loss_curves(train_losses, val_losses, num_epochs)

    torch.save(trained_model.state_dict(), 'trained_model.pth')

if __name__ == "__main__":
    main()
