import torch
import logging
import matplotlib.pyplot as plt
import torch.nn.functional as F


def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=1).sum(dim=1)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=1).sum(dim=1) + target.sum(dim=1).sum(dim=1) + smooth)))

    return loss.mean()

def calc_loss(pred, target, metrics, bce_weight=0.5):
    try:
        pred = torch.squeeze(pred, 1)
        bce = F.binary_cross_entropy_with_logits(pred, target)
        pred = torch.sigmoid(pred)
        dice = dice_loss(pred, target)
        loss = bce * bce_weight + dice * (1 - bce_weight)

        metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
        metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
        metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

        return loss
    except Exception as e:
        logging.error(f"Error calculating loss: {e}")
        raise

def print_metrics(metrics, epoch_samples, phase):
    try:
        outputs = [f"{k}: {v / epoch_samples:.4f}" for k, v in metrics.items()]
        logging.info(f"{phase}: {', '.join(outputs)}")
    except Exception as e:
        logging.error(f"Error printing metrics: {e}")
        raise

def plot_loss_curves(train_losses, val_losses, num_epochs):
    plt.plot(range(num_epochs), train_losses, label='Training Loss')
    plt.plot(range(num_epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.show()

def viz_from_dataloader(train_loader):
    import numpy as np
    import matplotlib.pyplot as plt
    vizdict = next(iter(train_loader))
    vizimg, vizmask = vizdict['X'][0], vizdict['y'][0]
    # Convert tensors to numpy arrays
    vizimg = vizimg.permute(1, 2, 0).numpy()
    vizmask = vizmask.squeeze().numpy()
    # Apply the mask on the image
    segmented_image = np.copy(vizimg)
    segmented_image[vizmask > 0] = [255, 0, 0]  # Set flooded areas to red
    # Plot original image and segmented image
    fig, axs = plt.subplots(1, 3, figsize=(6, 3))
    axs[0].imshow(vizimg)
    axs[0].set_title('Original Image')
    axs[1].imshow(vizmask, cmap='gray')
    axs[1].set_title('Mask Only')
    axs[2].imshow(segmented_image)
    axs[2].set_title('Segmented Image')
    # plt.show()
    plt.savefig('segmented_image_plot.png', dpi=300, bbox_inches='tight')
