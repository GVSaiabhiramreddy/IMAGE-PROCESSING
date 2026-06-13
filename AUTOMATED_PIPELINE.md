# Automated Training Pipeline - Quick Start

## One-Command Training

Run the complete automated training pipeline with a single command:

### Linux/Mac
```bash
bash run_full_pipeline.sh
```

### Windows
```batch
run_full_pipeline.bat
```

### Python (Cross-platform)
```bash
python run_automated_training.py
```

## What the Pipeline Does

1. **Create Sample Dataset** - Generates synthetic underwater images for demonstration
2. **Configure Training** - Sets up optimal parameters for quick training
3. **Install Dependencies** - Installs all required Python packages
4. **Train Model** - Executes the training loop with monitoring
5. **Run Inference** - Tests the trained model on test data
6. **Generate Report** - Creates comprehensive training report
7. **Organize Results** - Collects all outputs in a structured directory

## Output

All results are organized in `TRAINING_RESULTS/` directory:

```
TRAINING_RESULTS/
├── checkpoints/          # Trained model weights (*.pth files)
├── inference_results/    # Restored underwater images
├── logs/                 # Training and validation logs
└── REPORT.json          # Comprehensive training report
```

## Performance Expectations

- **Execution Time**: ~5-15 minutes (depending on hardware)
- **GPU Memory**: ~2-4GB (adjustable via batch size)
- **Model Accuracy**: Improves with epochs (demo: 10 epochs)

## Features Included

✓ Automatic dependency installation  
✓ Sample dataset generation  
✓ Multi-GPU support detection  
✓ Real-time training monitoring  
✓ Automatic inference on test data  
✓ Comprehensive reporting  
✓ Result organization  
✓ Error handling and recovery  

## Advanced Options

### Use Your Own Dataset

1. Place your data in:
   ```
   UIR-PolyKernel/data/
   ├── train/
   │   ├── input/  (underwater images)
   │   └── target/ (reference images)
   ├── val/
   │   ├── input/
   │   └── target/
   └── test/
       ├── input/
       └── target/
   ```

2. Update `UIR-PolyKernel/config.yml` with correct paths

3. Run pipeline:
   ```bash
   python run_automated_training.py
   ```

### Customize Training Parameters

Edit `config.yml` before running:

```yaml
OPTIM:
  BATCH_SIZE: 16          # Increase for better GPU utilization
  NUM_EPOCHS: 100         # More epochs = better accuracy
  LR_INITIAL: 2e-4        # Learning rate
```

## Troubleshooting

### Out of Memory
```yaml
# In config.yml, reduce:
OPTIM:
  BATCH_SIZE: 2  # Was 4
```

### Slow Training
```yaml
# Check GPU is being used - should see GPU memory usage
# If slow, ensure CUDA/GPU is properly installed
```

### Missing Dependencies
```bash
# Manually install
pip install -r UIR-PolyKernel/requirements.txt
```

## Results Interpretation

The `REPORT.json` contains:

- **PSNR**: Peak Signal-to-Noise Ratio (higher is better)
- **SSIM**: Structural Similarity Index (higher is better)  
- **UCIQE**: Underwater image quality metric (higher is better)
- **Training Loss**: Overall training convergence
- **Validation Metrics**: Performance on held-out data

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
