import torch
import torch.nn as nn
import time
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import os
from .losses import CombinedLoss
from ..utils.metrics import MetricsCalculator

# ## TRAINING MANAGER ===================================================================>
class Trainer:
    """Handles model training with checkpointing and resuming"""
    
    def __init__(self, model, config, train_loader, val_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma
        )
        
        # Setup loss
        self.criterion = CombinedLoss(
            bce_weight=config.bce_weight,
            dice_weight=config.dice_weight
        )
        
        # Tracking variables
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.train_metrics_history = defaultdict(list)
        self.val_metrics_history = defaultdict(list)
        
        # Metrics calculator
        self.metrics_calc = MetricsCalculator()
        
        logging.info(f"Trainer initialized on device: {self.device}")
    
    def train(self, num_epochs=None):
        """Main training loop"""
        num_epochs = num_epochs or self.config.num_epochs
        
        # Resume if requested
        if self.config.resume_training and self.config.resume_checkpoint:
            self.load_checkpoint(self.config.resume_checkpoint)
        
        logging.info(f"Starting training from epoch {self.start_epoch} to {num_epochs}")
        
        for epoch in range(self.start_epoch, num_epochs):
            epoch_start = time.time()
            
            # Train phase
            train_loss, train_metrics = self._train_epoch(epoch)
            self.train_losses.append(train_loss)
            for k, v in train_metrics.items():
                self.train_metrics_history[k].append(v)
            
            # Validation phase
            val_loss, val_metrics = self._validate_epoch(epoch)
            self.val_losses.append(val_loss)
            for k, v in val_metrics.items():
                self.val_metrics_history[k].append(v)
            
            # Scheduler step
            self.scheduler.step()
            
            # Logging
            epoch_time = time.time() - epoch_start
            logging.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} - "
                f"Time: {epoch_time//60:.0f}m {epoch_time%60:.0f}s"
            )
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(
                    os.path.join(self.config.checkpoint_dir, 'best_model.pth'),
                    epoch, is_best=True
                )
                logging.info(f"Saved best model with loss {val_loss:.4f}")
            
            # Save last epoch
            self.save_checkpoint(
                os.path.join(self.config.checkpoint_dir, 'last_epoch.pth'),
                epoch, is_best=False
            )
            
            # Save plots every N epochs
            if (epoch + 1) % self.config.save_interval == 0:
                self._save_training_plots(epoch + 1)
        
        # Final plots
        self._save_training_plots(num_epochs, final=True)
        
        logging.info("Training completed!")
        
        return self.model, self.train_losses, self.val_losses
    
    def _train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        metrics = defaultdict(float)
        epoch_samples = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            inputs = batch['X'].to(self.device)
            targets = batch['y'].to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Loss
            loss = self.criterion(outputs, targets, metrics)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            epoch_samples += inputs.size(0)
            
            # Periodic logging
            if (batch_idx + 1) % 10 == 0:
                logging.info(
                    f"Epoch {epoch+1} - Batch {batch_idx+1}/{len(self.train_loader)} - "
                    f"Loss: {loss.item():.4f}"
                )
        
        # Calculate average metrics
        avg_metrics = {k: v / epoch_samples for k, v in metrics.items()}
        return avg_metrics['loss'], avg_metrics
    
    def _validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        metrics = defaultdict(float)
        epoch_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch['X'].to(self.device)
                targets = batch['y'].to(self.device)
                
                # Forward
                outputs = self.model(inputs)
                
                # Loss
                loss = self.criterion(outputs, targets, metrics)
                
                epoch_samples += inputs.size(0)
        
        # Calculate average metrics
        avg_metrics = {k: v / epoch_samples for k, v in metrics.items()}
        return avg_metrics['loss'], avg_metrics
    
    def save_checkpoint(self, filepath, epoch, is_best=False):
        """Save checkpoint with all training state"""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics_history': dict(self.train_metrics_history),
            'val_metrics_history': dict(self.val_metrics_history),
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        logging.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load checkpoint and resume training"""
        if not os.path.exists(filepath):
            logging.warning(f"Checkpoint {filepath} not found. Starting from scratch.")
            return
        
        logging.info(f"Loading checkpoint from {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_metrics_history = defaultdict(list, checkpoint['train_metrics_history'])
        self.val_metrics_history = defaultdict(list, checkpoint['val_metrics_history'])
        
        logging.info(f"Resumed from epoch {self.start_epoch}, best loss: {self.best_loss:.4f}")
    
    def _save_training_plots(self, epoch, final=False):
        """Save training plots"""
        suffix = 'final' if final else f'epoch_{epoch}'
        
        # Loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.config.plots_dir}/loss_{suffix}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Metrics plots
        for metric_name in self.train_metrics_history.keys():
            if metric_name == 'loss':
                continue
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.train_metrics_history[metric_name], label=f'Train {metric_name}')
            plt.plot(self.val_metrics_history[metric_name], label=f'Val {metric_name}')
            plt.xlabel('Epoch')
            plt.ylabel(metric_name.capitalize())
            plt.title(f'Training and Validation {metric_name.capitalize()}')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{self.config.plots_dir}/{metric_name}_{suffix}.png", 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        logging.info(f"Saved training plots for {suffix}")
