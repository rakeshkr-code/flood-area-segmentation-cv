import torch
import numpy as np

# ## METRICS CALCULATOR ================================================================>
class MetricsCalculator:
    """Calculate various segmentation metrics"""
    
    @staticmethod
    def dice_coefficient(pred, target, smooth=1.0):
        """Calculate Dice coefficient"""
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        
        return dice.item()
    
    @staticmethod
    def iou_score(pred, target, smooth=1.0):
        """Calculate IoU score"""
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        
        return iou.item()
    
    @staticmethod
    def pixel_accuracy(pred, target):
        """Calculate pixel accuracy"""
        pred = (pred > 0.5).float()
        correct = (pred == target).sum()
        total = target.numel()
        
        return (correct / total).item()
