@echo off
REM UIR-PolyKernel Training Script for Windows
REM This script automates the training process for underwater image restoration

setlocal enabledelayedexpansion

echo ========================================
echo   UIR-PolyKernel Training Launcher
echo ========================================
echo.

REM Check if config.yml exists
if not exist "config.yml" (
    echo ERROR: config.yml not found!
    echo Please configure config.yml with your dataset paths first.
    exit /b 1
)

echo Configuration file found: config.yml
echo.
echo Starting training...
echo.

REM Check for GPU
where nvidia-smi >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo NVIDIA GPU detected.
    echo Using GPU Training...
    echo.
    python train.py
) else (
    echo No NVIDIA GPU detected.
    echo Using CPU Training...
    echo.
    python train.py
)

echo.
echo ========================================
echo   Training Complete!
echo ========================================
