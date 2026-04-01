@echo off
echo ======================================================================
echo Debugging Branch Loading for Model 0011
echo ======================================================================
echo.

REM Change to the script directory
cd /d "%~dp0"

echo Activating conda environment...
call conda.bat activate vmtk_env

echo Running debug script...
python debug_branches.py

echo.
echo ======================================================================
echo Done!
echo ======================================================================
pause

