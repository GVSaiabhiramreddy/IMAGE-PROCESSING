#!/bin/bash

# Complete Automated Training Pipeline
# Executes the entire training, validation, and inference workflow

echo ""
echo "*************************************************************"
echo "*  UIR-PolyKernel Automated Training Pipeline"
echo "*  Underwater Image Restoration Model"
echo "*************************************************************"
echo ""

# Navigate to UIR-PolyKernel directory
cd UIR-PolyKernel || exit 1

# Run the automated training
python ../run_automated_training.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "Training pipeline completed successfully!"
    echo "Results are available in: TRAINING_RESULTS/"
else
    echo ""
    echo "Training pipeline failed with exit code: $exit_code"
fi

exit $exit_code
