@echo off
REM Visualize model 0011 as stEVE sees it (NO ROTATION)
REM This shows the processed OBJ mesh with centerlines WITHOUT rotation

echo ========================================
echo Visualizing Model 0011 - As stEVE Sees It
echo ========================================
echo.

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0
REM Get the project root (parent of vmr_processing_tools)
cd /d "%SCRIPT_DIR%.."
set PROJECT_ROOT=%CD%

echo Project root: %PROJECT_ROOT%
echo.

REM Activate conda environment
call conda activate vmtk_env
if errorlevel 1 (
    echo ERROR: Failed to activate vmtk_env
    echo Please make sure conda is installed and vmtk_env exists
    pause
    exit /b 1
)

REM Set default model path (dualdevicenav_format folder)
set MODEL_PATH=D:\vmr\vmr\0011_H_AO_H\dualdevicenav_format

REM Check if custom path is provided
if not "%~1"=="" set MODEL_PATH=%~1

echo Model path: %MODEL_PATH%
echo.
echo This visualization shows:
echo   - Processed OBJ mesh (scaled only, NO rotation)
echo   - Centerlines (scaled only, NO rotation)
echo   - Raw data before stEVE applies rotation_yzx_deg
echo.

REM Run the visualization script from project root
python "%PROJECT_ROOT%\vmr_processing_tools\visualize_as_steve_sees_it.py" "%MODEL_PATH%"

REM Deactivate conda environment
call conda deactivate

echo.
echo ========================================
echo Visualization Complete
echo ========================================
pause

