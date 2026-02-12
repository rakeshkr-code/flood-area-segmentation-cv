import torch
import torch.nn as nn
import torch.nn.functional as F

# ## LOSSE FUNCTIONS ===================================================================>
class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()
        
        intersection = (pred * target).sum(dim=2).sum(dim=1)
        dice_score = (2. * intersection + self.smooth) / \
                     (pred.sum(dim=2).sum(dim=1) + target.sum(dim=2).sum(dim=1) + self.smooth)
        
        return (1 - dice_score).mean()


class CombinedLoss(nn.Module):
    """Combined BCE and Dice loss"""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss()
    
    def forward(self, pred, target, metrics=None):
        pred = torch.squeeze(pred, 1)
        
        # BCE loss
        bce = F.binary_cross_entropy_with_logits(pred, target)
        
        # Dice loss
        pred_sigmoid = torch.sigmoid(pred)
        dice = self.dice_loss(pred_sigmoid, target)
        
        # Combined
        loss = self.bce_weight * bce + self.dice_weight * dice
        
        # Update metrics if provided
        if metrics is not None:
            metrics['bce'] += bce.item() * target.size(0)
            metrics['dice'] += dice.item() * target.size(0)
            metrics['loss'] += loss.item() * target.size(0)
        
        return loss

if __name__ == "__main__":
    """Test loss functions"""
    import torch
    
    # Create dummy data
    pred = torch.randn(4, 1, 256, 256)
    target = torch.randint(0, 2, (4, 256, 256)).float()
    
    # Test losses
    criterion = CombinedLoss()
    # metrics = {}
    metrics = {'bce': 0.0, 'dice': 0.0, 'loss': 0.0}
    # from collections import defaultdict
    # metrics = defaultdict(float)  # Auto-initializes missing keys to 0.0
    loss = criterion(pred, target, metrics)
    
    print(f"Combined Loss: {loss.item():.4f}")
    print(f"BCE: {metrics['bce']/4:.4f}")
    print(f"Dice: {metrics['dice']/4:.4f}")
    print("✓ Loss calculation test passed")