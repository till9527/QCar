@echo off
:: This batch file automates the launch sequence for the multi-agent simulation.

:: --- CONFIGURATION ---
:: Define the full path to the JSON file that initCars.py is supposed to create.
:: IMPORTANT: Make sure the username ('user' in this path) matches your Windows username.
set JSON_FILE="C:\Users\user\Documents\Quanser\libraries\resources\rt_models\MultiAgent\RobotAgents.json"


:: --- STEP 1: RUN SETUP SCRIPT ---
echo --- [1/4] Starting Environment Setup (initCars.py)... ---
:: The /wait flag is crucial. It forces the batch script to wait until initCars.py is finished.
start "Setup" /wait cmd /k python initCars.py


:: --- STEP 2: VERIFY SETUP SUCCESS ---
echo --- [2/4] Verifying setup success... ---

:: Check 1: Did the python script exit with an error code?
if %errorlevel% neq 0 (
    echo.
    echo [FATAL ERROR] The 'initCars.py' script failed to run correctly.
    echo Please check the "Setup" window for Python error messages.
    echo Halting execution.
    pause
    goto :EOF
)

:: Check 2: Was the required RobotAgents.json file created?
if not exist %JSON_FILE% (
    echo.
    echo [FATAL ERROR] The required %JSON_FILE% was not found.
    echo 'initCars.py' completed without error, but failed to create the agent configuration file.
    echo Halting execution.
    pause
    goto :EOF
)

echo     -> Setup successful. The RobotAgents.json file was found.
echo.
echo --- Pausing for 5 seconds to let the simulator settle... ---
TIMEOUT /T 5 >nul


:: --- STEP 3: LAUNCH BACKGROUND LOGIC ---
echo.
echo --- [3/4] Starting Environment Logic (Pedestrians, Traffic Lights)... ---
start "Environment Logic" cmd /k python environment_logic.py


:: --- STEP 4: LAUNCH VEHICLE CONTROL SCRIPTS ---
echo.
echo --- [4/4] Starting Vehicle Control Scripts... ---
start "Car 1 Control" cmd /k python vehicle_control.py
TIMEOUT /T 1 >nul
start "Car 2 Control" cmd /k python vehicle_control2.py

echo.
echo --- All scripts have been launched successfully. ---
pause