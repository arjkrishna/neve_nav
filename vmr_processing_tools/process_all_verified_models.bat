@echo off
REM Batch file to process all 44 verified VMR models
REM This calls the Python script which handles the batch processing

echo ======================================================================
echo Batch Processing All Verified VMR Models
echo ======================================================================
echo.

REM Activate conda environment
echo Activating conda environment...
call conda activate vmtk_env

echo.
echo Running batch processing script...
echo.

REM Run the Python script
python "%~dp0process_all_verified_models.py" %*

echo.
echo ======================================================================
echo Done!
echo ======================================================================
pause








