@echo off
REM Batch file to run DualDeviceNav format creation without PowerShell issues

REM Default VMR folder for testing
SET DEFAULT_VMR=D:\vmr\vmr\0011_H_AO_H

echo ======================================================================
echo DualDeviceNav Format Creation
echo ======================================================================
echo.

echo Activating conda environment...
call conda activate vmtk_env

echo.
echo Running DualDeviceNav format creation...
echo.

REM Use provided argument or default
if "%~1"=="" (
    echo Using default VMR folder: %DEFAULT_VMR%
    python "%~dp0create_dualdevicenav_format.py" "%DEFAULT_VMR%"
) else (
    echo Using VMR folder: %~1
    python "%~dp0create_dualdevicenav_format.py" "%~1"
)

echo.
echo ======================================================================
echo Done!
echo ======================================================================
pause

