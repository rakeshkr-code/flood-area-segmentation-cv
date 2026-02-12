# Flood Area Segmentation

Deep learning pipeline for flood area segmentation using ResNet-UNet architecture.

## Project Structure

```
flood-area-segmentation/
├── archive/                    # Your data (Image/, Mask/, metadata.csv)
├── src/
│   ├── config.py              # Configuration
│   ├── data/                  # Dataset and dataloader
│   ├── models/                # ResNet-UNet model
│   ├── training/              # Training loop and losses
│   ├── inference/             # Inference pipeline
│   └── utils/                 # Visualization and metrics
├── train.py                   # Training script
├── inference.py               # Inference script
└── requirements.txt
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Verify setup (optional)
python -m src.utils.visualization
python -m src.models.unet
```

## Usage

### Train
```bash
python train.py
```

Outputs:
- `checkpoints/best_model.pth` - Best model
- `checkpoints/last_epoch.pth` - For resuming
- `outputs/plots/` - Loss curves every 10 epochs
- `logs/training.log` - Training logs

### Resume Training
```python
# Edit src/config.py
resume_training = True
resume_checkpoint = 'checkpoints/last_epoch.pth'
```

### Inference
```bash
python inference.py
```

Predictions saved to `outputs/predictions/` with red overlay on flood areas.

## Configuration

Edit `src/config.py`:
```python
batch_size = 8
num_epochs = 50
learning_rate = 1e-3
train_img_size = (576, 576)
mask_alpha = 0.4  # Overlay transparency
```

## Features

- **Modular class-based design**
- **Auto train/test split** with metadata CSV generation
- **GPU training** with CUDA support
- **Resume from checkpoint** if training fails
- **Visualization** - plot images/masks by filename or from dataloader
- **Overlay predictions** with customizable transparency

## Model

- **Architecture**: U-Net with ResNet18 encoder
- **Loss**: Combined BCE + Dice (50/50)
- **Augmentation**: Flips, rotation, color jitter
- **Parameters**: ~16.5M

## Requirements

- Python 3.8+
- PyTorch 2.0+
- 8GB GPU VRAM (for batch_size=8)

## Troubleshooting

**WSL matplotlib warning**: Ignore it, plots save correctly
```bash
explorer.exe outputs/plots/  # View on Windows
```

**Out of memory**: Reduce `batch_size` in config.py

**Import errors**: Run from project root directory

## Data Format

```
archive/
├── Image/
│   ├── 1.jpg
│   └── 2.jpg
├── Mask/
│   ├── 1.jpg
│   └── 2.jpg
└── metadata.csv  # Columns: Image, Mask
```