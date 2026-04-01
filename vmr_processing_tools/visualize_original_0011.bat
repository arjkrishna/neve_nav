@echo off
REM Visualize model 0011 original data (before transformations)
REM This shows the original VTP mesh with extracted centerlines to verify extraction quality

echo ========================================
echo Visualizing Model 0011 - Original Data
echo ========================================
echo.

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0
REM Get the project root (parent of vmr_processing_tools)
cd /d "%SCRIPT_DIR%.."
set PROJECT_ROOT=%CD%

echo Project root: %PROJECT_ROOT%
echo.

REM Initialize conda (finds conda.bat automatically)
call conda.bat activate vmtk_env
if errorlevel 1 (
    echo ERROR: Failed to activate vmtk_env
    echo.
    echo Try running this first in a terminal:
    echo   conda activate vmtk_env
    echo   python vmr_processing_tools\visualize_original_vs_processed.py D:\vmr\vmr\0011_H_AO_H
    pause
    exit /b 1
)

REM Set default model path (VMR model root folder)
set MODEL_PATH=D:\vmr\vmr\0011_H_AO_H

REM Check if custom path is provided
if not "%~1"=="" set MODEL_PATH=%~1

REM Check for mode (original or compare)
set MODE=original
if not "%~2"=="" set MODE=%~2

echo Model path: %MODEL_PATH%
echo Mode: %MODE%
echo.
echo This visualization shows:
echo   - Original VTP mesh (untransformed)
echo   - Extracted centerlines (LPS coordinates)
echo   - Verifies extraction quality
echo.

REM Run the visualization script from project root
if "%MODE%"=="compare" (
    python "%PROJECT_ROOT%\vmr_processing_tools\visualize_original_vs_processed.py" "%MODEL_PATH%" compare
) else (
    python "%PROJECT_ROOT%\vmr_processing_tools\visualize_original_vs_processed.py" "%MODEL_PATH%"
)

REM Deactivate conda environment
call conda deactivate

echo.
echo ========================================
echo Visualization Complete
echo ========================================
pause

