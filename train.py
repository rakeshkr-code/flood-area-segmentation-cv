import torch
import time
import logging
from collections import defaultdict
import torch.nn.functional as F

from model import ResNetUNet
from utils import calc_loss, print_metrics

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(model, optimizer, scheduler, train_loader, val_loader, device, num_epochs=25):
    train_losses = []
    val_losses = []

    best_loss = 1e10

    for epoch in range(num_epochs):
        logging.info(f'Starting epoch {epoch}/{num_epochs - 1}')
        since = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            metrics = defaultdict(float)
            epoch_samples = 0

            for sample in dataloader:
                inputs, labels = sample['X'].to(device), sample['y'].to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            train_losses.append(epoch_loss) if phase == 'train' else val_losses.append(epoch_loss)

            if phase == 'train':
              scheduler.step()
              for param_group in optimizer.param_groups:
                  print("LR", param_group['lr'])

            if phase == 'val' and epoch_loss < best_loss:
                logging.info(f'Saving best model with loss {epoch_loss:.4f}')
                best_loss = epoch_loss
                torch.save(model.state_dict(), "best_model.pth")

        time_elapsed = time.time() - since
        logging.info(f'Epoch {epoch} complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    return model, train_losses, val_losses
