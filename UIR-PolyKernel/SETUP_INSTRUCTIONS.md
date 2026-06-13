# UIR-PolyKernel Setup & Training Instructions

## Requirements

Install the necessary dependencies:

```bash
pip install torch torchvision
pip install accelerate
pip install torchmetrics
pip install tqdm
pip install yacs
pip install einops
pip install timm
pip install mmcv
pip install opencv-python
pip install albumentations
pip install numpy
pip install Pillow
pip install kornia
pip install imbalanced-learn
pip install wandb  # Optional for wandb logging
```

## Dataset Preparation

Organize your dataset as follows:

```
dataset/
├── train/
│   ├── input/
│   │   └── *.jpg (underwater images)
│   └── target/
│       └── *.jpg (reference/restored images)
├── val/
│   ├── input/
│   │   └── *.jpg
│   └── target/
│       └── *.jpg
└── test/
    ├── input/
    │   └── *.jpg
    └── target/
        └── *.jpg
```

## Configuration

Edit `config.yml` to set your data paths:

```yaml
TRAINING:
  TRAIN_DIR: '/path/to/dataset/train'
  VAL_DIR: '/path/to/dataset/val'
  SAVE_DIR: './checkpoints'

TESTING:
  VAL_DIR: '/path/to/dataset/test'
  WEIGHT: './checkpoints/UIR_PolyKernel_epoch_XXX.pth'
  RESULT_DIR: './results'

LOG:
  LOG_DIR: './logs'
```

## Training

### Single GPU Training

```bash
python train.py
```

### Multi-GPU Training

```bash
acccelerate config  # Configure accelerate
acccelerate launch train.py
```

## Testing/Inference

```bash
python test.py
```

Make sure to set the correct weight path in `config.yml` before running inference.

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
