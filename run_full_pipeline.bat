@echo off
REM Complete Automated Training Pipeline for Windows
REM Executes the entire training, validation, and inference workflow

echo.
echo *************************************************************
echo *  UIR-PolyKernel Automated Training Pipeline
echo *  Underwater Image Restoration Model
echo *************************************************************
echo.

REM Navigate to UIR-PolyKernel directory
cd UIR-PolyKernel
if errorlevel 1 exit /b 1

REM Run the automated training
python ../run_automated_training.py

set exit_code=%ERRORLEVEL%

if %exit_code% equ 0 (
    echo.
    echo Training pipeline completed successfully!
    echo Results are available in: TRAINING_RESULTS ^
    echo.
) else (
    echo.
    echo Training pipeline failed with exit code: %exit_code%
    echo.
)

exit /b %exit_code%
