import sys
import time
from qvl.multi_agent import MultiAgent
from qvl.qlabs import QuanserInteractiveLabs
from qvl.real_time import QLabsRealTime
from qvl.free_camera import QLabsFreeCamera

# --- Main Script Execution ---

qlabs = None
try:
    # 1. Connect to QLabs ONCE at the beginning
    print("Connecting to QLabs...")
    qlabs = QuanserInteractiveLabs()
    if not qlabs.open("localhost"):
        print("Unable to connect to QLabs")
        sys.exit()
    print("✅ Connected to QLabs.")

    # 2. Clean up the environment using the active connection
    print("Cleaning up existing actors and models...")
    QLabsRealTime().terminate_all_real_time_models()
    time.sleep(1)
    qlabs.destroy_all_spawned_actors()
    print("✅ Environment is clean.")

    # 3. Define the cars to be spawned
    QCars_to_spawn = [
        {
            "RobotType": "QC2",
            "Location": [-12.82, -4.60, 0],
            "Rotation": [0, 0, -0.733],
            "Radians": True,
        },
        {
            "RobotType": "QCar2",
            "Location": [22.55, 0.81, 0],
            "Rotation": [0, 0, 1.57],
            "Radians": True,
        },
    ]

    # 4. Use MultiAgent to spawn the cars AND create RobotAgents.json
    #    This requires the active qlabs connection.
    print("Spawning cars and creating configuration file...")
    MultiAgent(QCars_to_spawn)
    print("✅ Cars spawned and RobotAgents.json created.")

    # 5. Set up a camera view (optional, but uses the same connection)
    camera = QLabsFreeCamera(qlabs)
    camera.spawn_degrees(location=[28.00, -11.69, 33.17], rotation=[0, 51.4, 141.5])
    camera.possess()
    print("✅ Camera view set.")


except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # 6. Close the single QLabs connection at the very end
    if qlabs:
        qlabs.close()
        print("✅ Disconnected from QLabs. Setup is complete.")
