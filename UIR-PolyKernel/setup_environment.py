#!/usr/bin/env python3
"""
UIR-PolyKernel Environment Setup
Automatically installs dependencies and configures the environment
"""

import os
import sys
import subprocess
import platform
import yaml
from pathlib import Path

def print_header(text):
    print("\n" + "="*50)
    print(f"  {text}")
    print("="*50 + "\n")

def install_requirements():
    """Install Python dependencies from requirements.txt"""
    print_header("Installing Dependencies")
    
    requirements_file = "requirements.txt"
    if not os.path.exists(requirements_file):
        print(f"ERROR: {requirements_file} not found!")
        return False
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print("\n✓ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories for training"""
    print_header("Creating Directories")
    
    directories = [
        "checkpoints",
        "logs",
        "results",
        "data/train/input",
        "data/train/target",
        "data/val/input",
        "data/val/target",
        "data/test/input",
        "data/test/target"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    print("\n✓ All directories created successfully!")
    return True

def validate_config():
    """Validate configuration file"""
    print_header("Validating Configuration")
    
    config_file = "config.yml"
    if not os.path.exists(config_file):
        print(f"WARNING: {config_file} not found!")
        return False
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        print("Configuration loaded:")
        print(f"  Model Session: {config['MODEL']['SESSION']}")
        print(f"  Batch Size: {config['OPTIM']['BATCH_SIZE']}")
        print(f"  Number of Epochs: {config['OPTIM']['NUM_EPOCHS']}")
        print(f"  Learning Rate: {config['OPTIM']['LR_INITIAL']}")
        print("\n✓ Configuration is valid!")
        return True
    except Exception as e:
        print(f"✗ Error reading configuration: {e}")
        return False

def check_cuda():
    """Check if CUDA is available"""
    print_header("Checking CUDA Availability")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print(f"✓ CUDA is available!")
            print(f"  GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("⚠ CUDA is not available. CPU training will be used.")
            return False
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False

def main():
    """Main setup function"""
    print("\n" + "*"*50)
    print("*  UIR-PolyKernel Environment Setup")
    print("*"*50)
    
    print(f"\nPython Version: {sys.version}")
    print(f"Platform: {platform.system()} {platform.release()}")
    
    # Step 1: Install requirements
    if not install_requirements():
        print("\n✗ Setup failed at requirements installation.")
        return False
    
    # Step 2: Create directories
    if not create_directories():
        print("\n✗ Setup failed at directory creation.")
        return False
    
    # Step 3: Validate configuration
    validate_config()
    
    # Step 4: Check CUDA
    check_cuda()
    
    print_header("Setup Complete!")
    print("\nNext steps:")
    print("1. Configure your dataset paths in config.yml")
    print("2. Place your training data in the data/ directory")
    print("3. Run: python train.py (for single GPU)")
    print("4. Or run: bash run_training.sh (with automatic GPU detection)")
    print()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
