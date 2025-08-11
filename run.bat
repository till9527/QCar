@echo off
:: For this to run, open first Quanser Interactive labs, self driving car studio/CityScape or CityScape Lite.

:: 1. Run the setup script. It will spawn all actors and then exit.
echo --- Starting Environment Setup ---
start "Setup" /wait cmd /c python initCars.py

:: 2. Wait for a few seconds to ensure all actors are settled in the simulation.
echo --- Setup Complete. Pausing before starting logic... ---
TIMEOUT /T 5

:: 3. Start the persistent environment logic in a new background window.
echo --- Starting Environment Logic (Pedestrians, Traffic Lights) ---
start "Environment Logic" cmd /k python environment_logic.py

:: 4. Start the vehicle control scripts.
echo --- Starting Vehicle Control ---
start "Car 1 Control" cmd /k python vehicle_control.py
TIMEOUT /T 1
start "Car 2 Control" cmd /k python vehicle_control2.py

echo --- All scripts launched. ---
cmd /k