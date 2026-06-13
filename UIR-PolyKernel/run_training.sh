#!/bin/bash

# UIR-PolyKernel Training Script
# This script automates the training process for underwater image restoration

set -e  # Exit on error

echo "========================================"
echo "  UIR-PolyKernel Training Launcher"
echo "========================================"
echo ""

# Check if config.yml exists
if [ ! -f "config.yml" ]; then
    echo "ERROR: config.yml not found!"
    echo "Please configure config.yml with your dataset paths first."
    exit 1
fi

# Extract paths from config.yml
TRAIN_DIR=$(grep -A 5 "TRAINING:" config.yml | grep "TRAIN_DIR:" | awk -F"'" '{print $2}')
VAL_DIR=$(grep -A 5 "TRAINING:" config.yml | grep "VAL_DIR:" | awk -F"'" '{print $2}')
SAVE_DIR=$(grep -A 5 "TRAINING:" config.yml | grep "SAVE_DIR:" | awk -F"'" '{print $2}')

echo "Configuration Summary:"
echo "  Training Data: $TRAIN_DIR"
echo "  Validation Data: $VAL_DIR"
echo "  Checkpoint Save: $SAVE_DIR"
echo ""

# Validate paths
if [ -z "$TRAIN_DIR" ] || [ -z "$VAL_DIR" ] || [ -z "$SAVE_DIR" ]; then
    echo "ERROR: One or more paths are empty in config.yml"
    exit 1
fi

if [ ! -d "$TRAIN_DIR" ]; then
    echo "ERROR: Training directory not found: $TRAIN_DIR"
    exit 1
fi

if [ ! -d "$VAL_DIR" ]; then
    echo "ERROR: Validation directory not found: $VAL_DIR"
    exit 1
fi

# Create save directory if it doesn't exist
mkdir -p "$SAVE_DIR"
mkdir -p "logs"

echo "Starting training..."
echo ""

# Check if using multi-GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "GPUs detected: $GPU_COUNT"
    
    if [ $GPU_COUNT -gt 1 ]; then
        echo "Using Multi-GPU Training with Accelerate..."
        echo ""
        accelerate launch --multi_gpu train.py
    else
        echo "Using Single GPU Training..."
        echo ""
        python train.py
    fi
else
    echo "No NVIDIA GPUs detected. Using CPU training..."
    python train.py
fi

echo ""
echo "========================================"
echo "  Training Complete!"
echo "========================================"
echo "Checkpoints saved to: $SAVE_DIR"
