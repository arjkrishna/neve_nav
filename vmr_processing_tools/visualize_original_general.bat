@echo off
REM Visualize original data - Generic Model
REM This shows the original VTP mesh with extracted centerlines to verify extraction quality
REM
REM Usage:
REM   visualize_original_general.bat 0011
REM   visualize_original_general.bat 0011_H_AO_H
REM   visualize_original_general.bat 0011 compare

echo ========================================
echo Visualize Original Data - Generic Model
echo ========================================
echo.

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0
REM Get the project root (parent of vmr_processing_tools)
cd /d "%SCRIPT_DIR%.."
set PROJECT_ROOT=%CD%

echo Project root: %PROJECT_ROOT%
echo.

REM Store command line arguments BEFORE setlocal
set ARG1=%~1
set ARG2=%~2

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
echo   - Original VTP mesh (untransformed)
echo   - Extracted centerlines (LPS coordinates)
echo   - Verifies extraction quality
echo.
echo Press Ctrl+C to quit at any time
echo.

REM Check for mode once at the start (can be overridden by argument)
if not "!ARG2!"=="" (
    set MODE=!ARG2!
) else (
    echo Select visualization mode ^(will be used for all models^):
    echo   1. original - Show original VTP with centerlines ^(default^)
    echo   2. compare  - Side-by-side comparison of original VTP vs processed OBJ
    set /p MODE_INPUT="Enter mode (1 or 2, default=1): "
    if "!MODE_INPUT!"=="2" (
        set MODE=compare
    ) else (
        set MODE=original
    )
    echo.
)

REM Track if we've used the first argument
set FIRST_ARG_USED=0

REM Loop to visualize multiple models
:VISUALIZE_LOOP

REM Check if model name is provided as argument (first iteration only), otherwise prompt
if "!FIRST_ARG_USED!"=="0" (
    if not "!ARG1!"=="" (
        set MODEL_NAME=!ARG1!
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
echo Mode: !MODE!
echo.

REM Run the visualization script from project root
python "%PROJECT_ROOT%\vmr_processing_tools\visualize_original_general.py" --model_name !MODEL_NAME! --mode !MODE!

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

