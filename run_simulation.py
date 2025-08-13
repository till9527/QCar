import time
import signal
from threading import Thread
import multiprocessing as mp
import cv2

# Your existing modules
import perception_module
from vehicle_control import controlLoop as control_loop_1
from vehicle_control2 import controlLoop as control_loop_2

# Quanser libraries for setup and control
from qvl.qlabs import QuanserInteractiveLabs
from qvl.multi_agent import MultiAgent, readRobots
from qvl.free_camera import QLabsFreeCamera
from qvl.real_time import QLabsRealTime

# This global list will hold our threads for cleanup
threads = []

# --- Main Setup and Execution ---
if __name__ == '__main__':
    # Queues for perception images
    image_queue_0 = mp.Queue(maxsize=5)
    image_queue_1 = mp.Queue(maxsize=5)
    qlabs = None

    # Graceful shutdown handler
    def cleanup():
        global qlabs, threads
        print("\nShutdown signal received...")
        perception_module.KILL_THREAD = True
        for t in threads:
            if t.is_alive():
                t.join(timeout=1)
        if qlabs:
            qlabs.close()
            print("✅ QLabs connection closed.")
        cv2.destroyAllWindows()
        print("✅ All processes terminated.")

    signal.signal(signal.SIGINT, lambda sig, frame: cleanup())

    try:
        # === Step 1: Connect, Clean, Spawn Cars, and READ CONFIG ===
        print("Connecting to QLabs and setting up environment...")
        qlabs = QuanserInteractiveLabs()
        qlabs.open("localhost")
        QLabsRealTime().terminate_all_real_time_models()
        time.sleep(1)
        qlabs.destroy_all_spawned_actors()
        QCars_to_spawn = [
            {"RobotType": "QCar2", "Location": [-12.82, -4.60, 0], "Rotation": [0, 0, -0.733], 'Radians': True},
            {"RobotType": "QC2", "Location": [22.55, 0.81, 0], "Rotation": [0, 0, 1.57], 'Radians': True}
        ]
        MultiAgent(QCars_to_spawn)
        print("✅ Cars spawned.")

        robot_configs = readRobots()
        car_0_config = robot_configs["QC2_0"]
        car_1_config = robot_configs["QC2_1"]
        print("✅ Robot configuration loaded.")
        
        camera = QLabsFreeCamera(qlabs)
        camera.spawn_degrees(location=[28.00, -11.69, 33.17], rotation=[0, 51.4, 141.5])
        camera.possess()
        print("✅ Environment setup complete.")

        # === THE FIX: Add a pause to let the HIL servers initialize ===
        print("Pausing for HIL servers to start...")
        time.sleep(3) # A 3-second delay is usually sufficient

        # === Step 2: Launch All Background Threads ===
        threads.extend([
            Thread(target=control_loop_1),
            Thread(target=control_loop_2),
            Thread(target=perception_module.run_perception, args=(0, qlabs, image_queue_0)),
            Thread(target=perception_module.run_perception, args=(1, qlabs, image_queue_1))
        ])
        for t in threads:
            t.start()
        print("✅ All systems are running. Press 'q' in a detection window to exit.")

        # === Step 3: Main Display Loop ===
        while True:
            if not image_queue_0.empty():
                cv2.imshow("YOLO Detection - Car 0", image_queue_0.get())
            if not image_queue_1.empty():
                cv2.imshow("YOLO Detection - Car 1", image_queue_1.get())
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cleanup()

    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        print(f"An unhandled exception occurred: {e}")
        cleanup()