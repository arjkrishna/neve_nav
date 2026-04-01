@echo off
REM Batch file to visualize all verified models - As stEVE Sees It

echo ======================================================================
echo Visualize All Verified Models - As stEVE Sees It
echo ======================================================================
echo.

REM Activate conda environment
echo Activating conda environment...
call conda activate vmtk_env

echo.
echo Running visualization script...
echo.

REM Run the Python script
python "%~dp0visualize_all_models_as_steve.py" %*

echo.
echo ======================================================================
echo Done!
echo ======================================================================
pause








