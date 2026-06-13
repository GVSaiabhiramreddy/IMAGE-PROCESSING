# UIR-PolyKernel Training Guide

## Quick Start

### 1. Environment Setup

```bash
python setup_environment.py
```

This will:
- Install all required dependencies
- Create necessary directories
- Validate your configuration
- Check CUDA availability

### 2. Prepare Your Dataset

Organize your underwater image dataset:

```
data/
├── train/
│   ├── input/      # Underwater images
│   └── target/     # Reference/clear images
├── val/
│   ├── input/
│   └── target/
└── test/
    ├── input/
    └── target/
```

Supported formats: `.jpg`, `.png`, `.jpeg`

### 3. Configure Training Parameters

Edit `config.yml`:

```yaml
TRAINING:
  TRAIN_DIR: 'data/train'      # Path to training data
  VAL_DIR: 'data/val'          # Path to validation data
  SAVE_DIR: 'checkpoints'      # Where to save models
  BATCH_SIZE: 16               # Batch size (adjust for GPU memory)
  NUM_EPOCHS: 500              # Number of training epochs

LOG:
  LOG_DIR: 'logs'              # Where to save logs
```

### 4. Start Training

#### Option A: Automatic (Recommended)
```bash
# Linux/Mac
bash run_training.sh

# Windows
run_training.bat
```

#### Option B: Single GPU
```bash
python train.py
```

#### Option C: Multi-GPU
```bash
acccelerate config  # Configure accelerate (first time only)
acccelerate launch train.py
```

## Training Parameters

### Model Configuration
- **SESSION**: Model name for checkpoints
- **INPUT**: Input folder name
- **TARGET**: Target folder name

### Optimization
- **BATCH_SIZE**: 16 (default). Reduce if out of memory, increase for better results
- **NUM_EPOCHS**: 500 (default). More epochs = better results but longer training
- **LR_INITIAL**: 2e-4 (learning rate start)
- **LR_MIN**: 1e-6 (minimum learning rate)
- **SEED**: 3407 (for reproducibility)

### Training Details
- **PATCH_SIZE**: 256x256 (default). Larger patches need more GPU memory
- **VAL_AFTER_EVERY**: 1 (validate every N epochs)
- **WANDB**: False (enable for experiment tracking)

## Loss Functions

The model uses a combined loss:
- **L_total = L_PSNR + 0.2 × L_SSIM + 0.01 × L_UCIQE**
  - L_PSNR: Peak Signal-to-Noise Ratio (SmoothL1)
  - L_SSIM: Structural Similarity Index
  - L_UCIQE: Underwater Color Image Quality Evaluation

## Monitoring Training

### Real-time Metrics
- **PSNR**: Peak Signal-to-Noise Ratio (higher is better)
- **SSIM**: Structural Similarity Index (higher is better)
- **UCIQE**: Underwater image quality (higher is better)

### Logs
Training logs are saved to `logs/` directory.

### Checkpoints
Model checkpoints are saved to `checkpoints/` directory as:
```
UIR_PolyKernel_epoch_10.pth
UIR_PolyKernel_epoch_20.pth
...
```

## Performance Tips

### GPU Memory
- Reduce BATCH_SIZE if getting out-of-memory errors
- Reduce PATCH_SIZE (PS_W, PS_H) for lower GPU memory
- Use torch.cuda.empty_cache() between batches

### Training Speed
- Increase BATCH_SIZE for better GPU utilization
- Use multi-GPU training with accelerate
- Ensure num_workers is appropriate for your system

### Model Quality
- Train for more epochs (500-1000)
- Use lower learning rates for fine-tuning
- Augment your dataset with more variations
- Validate frequently to catch overfitting

## Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size in config.yml
BATCH_SIZE: 8  # Instead of 16

# Or reduce patch size
PS_W: 128
PS_H: 128
```

### Slow Training
- Check GPU utilization: `nvidia-smi`
- Increase num_workers in train.py
- Use multi-GPU training

### NaN Loss
- Reduce learning rate
- Check data normalization (should be 0-1)
- Increase batch size

## Testing/Inference

```bash
python test.py
```

Make sure to set in config.yml:
```yaml
TESTING:
  VAL_DIR: 'data/test'
  WEIGHT: 'checkpoints/UIR_PolyKernel_epoch_100.pth'
  RESULT_DIR: 'results'
```

## Citation

```bibtex
@inproceedings{guo2025underwater,
  title={Underwater Image Restoration via Polymorphic Large Kernel CNNs},
  author={Guo, Xiaojiao and Dong, Yihang and Chen, Xuhang and Chen, Weiwen and Li, Zimeng and Zheng, FuChen and Pun, Chi-Man},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```

## References

- [Original Repository](https://github.com/CXH-Research/UIR-PolyKernel)
- [ICASSP 2025 Paper](https://arxiv.org/abs/2412.18459)
- [Accelerate Documentation](https://huggingface.co/docs/accelerate)
