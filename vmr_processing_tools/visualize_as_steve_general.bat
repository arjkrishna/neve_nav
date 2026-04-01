@echo off
REM Visualize processed data as stEVE sees it - Generic Model
REM This shows the processed OBJ mesh with centerlines WITHOUT rotation
REM
REM Usage:
REM   visualize_as_steve_general.bat 0011
REM   visualize_as_steve_general.bat 0011_H_AO_H

echo ========================================
echo Visualize As stEVE Sees It - Generic Model
echo ========================================
echo.

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0
REM Get the project root (parent of vmr_processing_tools)
cd /d "%SCRIPT_DIR%.."
set PROJECT_ROOT=%CD%

echo Project root: %PROJECT_ROOT%
echo.

REM Enable delayed expansion for variables in if blocks
setlocal enabledelayedexpansion

REM Activate conda environment (once at the start)
call conda activate vmtk_env
if errorlevel 1 (
    echo ERROR: Failed to activate vmtk_env
    echo Please make sure conda is installed and vmtk_env exists
    pause
    exit /b 1
)

echo This visualization shows:
echo   - Processed OBJ mesh (scaled only, NO rotation)
echo   - Centerlines (scaled only, NO rotation)
echo   - Raw data before stEVE applies rotation_yzx_deg
echo.
echo Press Ctrl+C to quit at any time
echo.

REM Track if we've used the first argument
set FIRST_ARG_USED=0

REM Loop to visualize multiple models
:VISUALIZE_LOOP

REM Check if model name is provided as argument (first iteration only), otherwise prompt
if "!FIRST_ARG_USED!"=="0" (
    if not "%~1"=="" (
        set MODEL_NAME=%~1
        set FIRST_ARG_USED=1
    ) else (
        echo.
        set /p MODEL_NAME="Enter model number or name (e.g., 0011 or 0011_H_AO_H, or 'q' to quit): "
        if "!MODEL_NAME!"=="" (
            echo Exiting...
            goto :END
        )
        if /i "!MODEL_NAME!"=="q" (
            echo Exiting...
            goto :END
        )
    )
) else (
    echo.
    set /p MODEL_NAME="Enter model number or name (e.g., 0011 or 0011_H_AO_H, or 'q' to quit): "
    if "!MODEL_NAME!"=="" (
        echo Exiting...
        goto :END
    )
    if /i "!MODEL_NAME!"=="q" (
        echo Exiting...
        goto :END
    )
)

echo.
echo Model name: !MODEL_NAME!
echo.

REM Run the visualization script from project root
python "%PROJECT_ROOT%\vmr_processing_tools\visualize_as_steve_general.py" --model_name !MODEL_NAME!

echo.
echo ========================================
echo Visualization Complete
echo ========================================
echo.

REM Continue to next model
goto :VISUALIZE_LOOP

:END
REM Deactivate conda environment
call conda deactivate

echo.
echo ========================================
echo All visualizations complete
echo ========================================
pause

