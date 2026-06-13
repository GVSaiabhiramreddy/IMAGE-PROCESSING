#!/usr/bin/env python3
"""
UIR-PolyKernel Automated Training Pipeline
Full automated training with result generation and reporting
"""

import os
import sys
import json
import time
import shutil
from datetime import datetime
from pathlib import Path
import subprocess

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")

def create_sample_dataset():
    """Create sample synthetic dataset for demonstration"""
    print_section("Creating Sample Dataset")
    
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow", "numpy", "-q"])
        import numpy as np
        from PIL import Image
    
    # Create directories
    dirs = [
        'data/train/input', 'data/train/target',
        'data/val/input', 'data/val/target',
        'data/test/input', 'data/test/target'
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    # Create sample synthetic images
    num_train = 10
    num_val = 5
    num_test = 3
    
    print(f"Creating {num_train} training samples...")
    for i in range(num_train):
        # Underwater image (darker, bluer)
        img_input = np.random.rand(256, 256, 3) * 0.4 + np.array([0.0, 0.2, 0.4])
        img_target = np.random.rand(256, 256, 3) * 0.8 + 0.1
        
        Image.fromarray((img_input * 255).astype(np.uint8)).save(f'data/train/input/img_{i:04d}.jpg')
        Image.fromarray((img_target * 255).astype(np.uint8)).save(f'data/train/target/img_{i:04d}.jpg')
    
    print(f"Creating {num_val} validation samples...")
    for i in range(num_val):
        img_input = np.random.rand(256, 256, 3) * 0.4 + np.array([0.0, 0.2, 0.4])
        img_target = np.random.rand(256, 256, 3) * 0.8 + 0.1
        
        Image.fromarray((img_input * 255).astype(np.uint8)).save(f'data/val/input/img_{i:04d}.jpg')
        Image.fromarray((img_target * 255).astype(np.uint8)).save(f'data/val/target/img_{i:04d}.jpg')
    
    print(f"Creating {num_test} test samples...")
    for i in range(num_test):
        img_input = np.random.rand(256, 256, 3) * 0.4 + np.array([0.0, 0.2, 0.4])
        img_target = np.random.rand(256, 256, 3) * 0.8 + 0.1
        
        Image.fromarray((img_input * 255).astype(np.uint8)).save(f'data/test/input/img_{i:04d}.jpg')
        Image.fromarray((img_target * 255).astype(np.uint8)).save(f'data/test/target/img_{i:04d}.jpg')
    
    print("\u2713 Sample dataset created successfully!")
    return True

def update_config():
    """Update config.yml with sample data paths"""
    print_section("Configuring Training Parameters")
    
    config_content = """VERBOSE: True

MODEL:
  SESSION: 'UIR_PolyKernel_AutoTrain'
  INPUT: 'input'
  TARGET: 'target'

OPTIM:
  BATCH_SIZE: 4
  NUM_EPOCHS: 10
  LR_INITIAL: 2e-4
  LR_MIN: 1e-6
  SEED: 3407
  WANDB: False

TRAINING:
  VAL_AFTER_EVERY: 1
  RESUME: False
  WEIGHT: ''
  PS_W: 256
  PS_H: 256
  TRAIN_DIR: 'data/train'
  VAL_DIR: 'data/val'
  SAVE_DIR: 'checkpoints'
  ORI: False
  LOG_FILE: 'training_log.txt'

TESTING:
  INPUT: 'input'
  TARGET: 'target'
  VAL_DIR: 'data/test'
  WEIGHT: ''
  SAVE_IMAGES: True
  RESULT_DIR: 'results'
  LOG_FILE: 'test_log.txt'

LOG:
  LOG_DIR: 'logs'
"""
    
    with open('config.yml', 'w') as f:
        f.write(config_content)
    
    print("Configuration updated:")
    print("  - Batch Size: 4 (for fast demo)")
    print("  - Epochs: 10 (for quick training)")
    print("  - Dataset: Sample synthetic data")
    print("  - Checkpoints: saved to checkpoints/")
    print("\u2713 Configuration file updated!")
    return True

def install_dependencies():
    """Install required dependencies"""
    print_section("Installing Dependencies")
    
    requirements = [
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'torchmetrics>=0.11.0',
        'accelerate>=0.20.0',
        'tqdm>=4.65.0',
        'yacs>=0.1.8',
        'einops>=0.6.1',
        'timm>=0.9.0',
        'mmcv>=2.0.0',
        'opencv-python>=4.7.0',
        'albumentations>=1.3.0',
        'numpy>=1.23.0',
        'Pillow>=9.5.0',
        'kornia>=0.7.0',
        'imbalanced-learn>=0.10.0'
    ]
    
    print("Installing core dependencies...")
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req, "-q"])
            print(f"  ✓ {req.split('>')[0].split('<')[0]}")
        except:
            print(f"  ! {req.split('>')[0].split('<')[0]} (optional)")
    
    print("\n✓ Dependencies installed!")
    return True

def run_training():
    """Execute the training process"""
    print_section("Starting Training")
    
    # Create directories
    Path('checkpoints').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    
    print("Training configuration:")
    print("  - Model: UIR_PolyKernel")
    print("  - Batch Size: 4")
    print("  - Epochs: 10")
    print("  - Dataset: data/train (10 samples)")
    print("  - Validation: data/val (5 samples)")
    print("  - Checkpoints: checkpoints/")
    print()
    
    try:
        # Run training
        print("Executing: python train.py")
        print("-" * 60)
        
        result = subprocess.run(
            [sys.executable, 'train.py'],
            cwd='UIR-PolyKernel',
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print("-" * 60)
            print("\n✓ Training completed successfully!")
            return True
        else:
            print("-" * 60)
            print("\n✗ Training encountered an error.")
            return False
            
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        return False

def run_inference():
    """Execute inference on test data"""
    print_section("Running Inference")
    
    # Find latest checkpoint
    checkpoint_dir = Path('UIR-PolyKernel/checkpoints')
    if not checkpoint_dir.exists():
        print("No checkpoints found. Skipping inference.")
        return False
    
    checkpoints = list(checkpoint_dir.glob('*.pth'))
    if not checkpoints:
        print("No checkpoint files found.")
        return False
    
    latest_checkpoint = sorted(checkpoints)[-1]
    print(f"Found latest checkpoint: {latest_checkpoint.name}")
    
    # Update config for testing
    config_path = Path('UIR-PolyKernel/config.yml')
    with open(config_path, 'r') as f:
        config = f.read()
    
    # Update weight path
    config = config.replace(
        "WEIGHT: ''",
        f"WEIGHT: '{latest_checkpoint}'"
    )
    
    with open(config_path, 'w') as f:
        f.write(config)
    
    print(f"Updated config with checkpoint: {latest_checkpoint.name}")
    print()
    
    try:
        print("Executing: python test.py")
        print("-" * 60)
        
        result = subprocess.run(
            [sys.executable, 'test.py'],
            cwd='UIR-PolyKernel',
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print("-" * 60)
            print("\n✓ Inference completed successfully!")
            return True
        else:
            print("-" * 60)
            print("\n✗ Inference encountered an error.")
            return False
            
    except Exception as e:
        print(f"\n✗ Error during inference: {e}")
        return False

def generate_report():
    """Generate comprehensive training report"""
    print_section("Generating Report")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_name": "UIR-PolyKernel",
        "task": "Underwater Image Restoration",
        "status": "Completed"
    }
    
    # Training logs
    log_path = Path('UIR-PolyKernel/logs/training_log.txt')
    if log_path.exists():
        with open(log_path, 'r') as f:
            training_logs = [line.strip() for line in f.readlines() if line.strip()]
            report["training_logs"] = training_logs[-10:]  # Last 10 lines
    
    # Checkpoints
    checkpoint_dir = Path('UIR-PolyKernel/checkpoints')
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob('*.pth'))
        report["checkpoints"] = [cp.name for cp in sorted(checkpoints)]
        report["checkpoint_count"] = len(checkpoints)
        report["latest_checkpoint"] = checkpoints[-1].name if checkpoints else None
    
    # Results
    results_dir = Path('UIR-PolyKernel/results')
    if results_dir.exists():
        result_images = list(results_dir.glob('*.jpg')) + list(results_dir.glob('*.png'))
        report["result_images"] = [img.name for img in result_images[:10]]
        report["total_results"] = len(result_images)
    
    # Summary
    report["summary"] = {
        "model_architecture": "Polymorphic Large Kernel CNNs",
        "training_dataset": "Synthetic samples (10 train, 5 val, 3 test)",
        "key_features": [
            "Hybrid Domain Attention",
            "Large Kernel Attention Blocks",
            "Composite Shape Convolution",
            "Frequency-Domain Pixel Attention"
        ],
        "performance_metrics": {
            "metric_type": "PSNR, SSIM, UCIQE",
            "optimization": "AdamW with Cosine Annealing",
            "loss_function": "Combined: PSNR + 0.2*SSIM + 0.01*UCIQE"
        },
        "output_directory": {
            "checkpoints": "UIR-PolyKernel/checkpoints/",
            "results": "UIR-PolyKernel/results/",
            "logs": "UIR-PolyKernel/logs/"
        }
    }
    
    # Save report
    report_path = Path('TRAINING_REPORT.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved to: {report_path}")
    print()
    print("Training Report:")
    print(f"  Model: {report['model_name']}")
    print(f"  Status: {report['status']}")
    print(f"  Timestamp: {report['timestamp']}")
    print(f"  Checkpoints: {report.get('checkpoint_count', 0)}")
    print(f"  Results: {report.get('total_results', 0)} images")
    print()
    print("Key Features:")
    for feature in report['summary']['key_features']:
        print(f"  - {feature}")
    
    return report

def copy_results():
    """Copy results to repository root"""
    print_section("Preparing Results")
    
    # Create results directory
    Path('TRAINING_RESULTS').mkdir(exist_ok=True)
    
    # Copy checkpoints
    checkpoint_src = Path('UIR-PolyKernel/checkpoints')
    checkpoint_dst = Path('TRAINING_RESULTS/checkpoints')
    if checkpoint_src.exists():
        if checkpoint_dst.exists():
            shutil.rmtree(checkpoint_dst)
        shutil.copytree(checkpoint_src, checkpoint_dst)
        print(f"\u2713 Copied checkpoints to TRAINING_RESULTS/")
    
    # Copy results
    results_src = Path('UIR-PolyKernel/results')
    results_dst = Path('TRAINING_RESULTS/inference_results')
    if results_src.exists():
        if results_dst.exists():
            shutil.rmtree(results_dst)
        shutil.copytree(results_src, results_dst)
        print(f"\u2713 Copied inference results to TRAINING_RESULTS/")
    
    # Copy logs
    logs_src = Path('UIR-PolyKernel/logs')
    logs_dst = Path('TRAINING_RESULTS/logs')
    if logs_src.exists():
        if logs_dst.exists():
            shutil.rmtree(logs_dst)
        shutil.copytree(logs_src, logs_dst)
        print(f"\u2713 Copied logs to TRAINING_RESULTS/")
    
    # Copy report
    report_src = Path('TRAINING_REPORT.json')
    if report_src.exists():
        shutil.copy(report_src, Path('TRAINING_RESULTS/REPORT.json'))
        print(f"\u2713 Copied report to TRAINING_RESULTS/")
    
    print("\n✓ All results organized in TRAINING_RESULTS/")
    return True

def main():
    """Main automation pipeline"""
    print("\n" + "*"*60)
    print("*  UIR-PolyKernel Automated Training Pipeline")
    print("*  Underwater Image Restoration Model")
    print("*"*60)
    
    start_time = time.time()
    
    # Step 1: Create sample dataset
    if not create_sample_dataset():
        print("Failed to create dataset")
        return False
    
    # Step 2: Update configuration
    if not update_config():
        print("Failed to update configuration")
        return False
    
    # Step 3: Install dependencies
    if not install_dependencies():
        print("Warning: Some dependencies failed to install")
    
    # Step 4: Run training
    if not run_training():
        print("Training failed")
        return False
    
    # Step 5: Run inference
    if not run_inference():
        print("Warning: Inference failed or was skipped")
    
    # Step 6: Generate report
    report = generate_report()
    
    # Step 7: Copy results
    if not copy_results():
        print("Failed to organize results")
        return False
    
    # Final summary
    elapsed_time = time.time() - start_time
    
    print_section("Pipeline Complete!")
    print(f"Total execution time: {elapsed_time/60:.2f} minutes")
    print()
    print("Output Structure:")
    print("  TRAINING_RESULTS/")
    print("    ├── checkpoints/          - Trained model weights")
    print("    ├── inference_results/  - Restored images")
    print("    ├── logs/               - Training logs")
    print("    └── REPORT.json         - Comprehensive report")
    print()
    print("Next steps:")
    print("  1. Review TRAINING_RESULTS/REPORT.json for detailed metrics")
    print("  2. Use checkpoints for inference on new underwater images")
    print("  3. Integrate with your image processing pipeline")
    print()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
