@echo off
REM ====================================================================
REM DualDeviceNav Format Creation (NO PRE-ROTATION)
REM ====================================================================
REM This batch file runs create_dualdevicenav_format_norot.py
REM 
REM Creates OBJ and JSON files WITHOUT rotation:
REM - OBJ mesh: Scaled only (cm -> mm), NO rotation
REM - JSON centerlines: Scaled only, NO rotation
REM - All rotation handled by stEVE via rotation_yzx_deg
REM
REM This is a CLEANER approach than pre-rotating the data!
REM ====================================================================

echo ======================================================================
echo DualDeviceNav Format Creation (NO PRE-ROTATION)
echo ======================================================================

REM Activate conda environment
echo Activating conda environment...
call conda activate vmtk_env

REM Default VMR folder
set DEFAULT_VMR=D:\vmr\vmr\0011_H_AO_H

REM Run the Python script
echo Running DualDeviceNav format creation (NO PRE-ROTATION)...
echo Using default VMR folder: %DEFAULT_VMR%
python "%~dp0create_dualdevicenav_format_norot.py" "%DEFAULT_VMR%"

echo.
echo ======================================================================
echo Done!
echo ======================================================================
pause









