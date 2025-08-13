import cv2
import torch
import sys
import time
from ultralytics import YOLO

# Quanser Imports
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar import QLabsQCar
from qvl.real_time import QLabsRealTime

# --- Configuration ---
MODEL_PATH = "model/best.pt"
CAMERA_TO_USE = QLabsQCar.CAMERA_RGB

if __name__ == "__main__":
    qlabs = None
    try:
        # --- 1. Connect to QLabs and Clean the Environment ---
        print("Connecting to QLabs...")
        qlabs = QuanserInteractiveLabs()
        if not qlabs.open("localhost"):
            print("FATAL: Unable to connect to QLabs. Is the simulation running?")
            sys.exit(1)
        print("✅ Connection successful!")

        print("Cleaning up existing actors...")
        QLabsRealTime().terminate_all_real_time_models()
        time.sleep(1)
        qlabs.destroy_all_spawned_actors()
        print("✅ Environment is clean.")

        # --- 2. Load the YOLO Model ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading YOLO model on device: '{device}'...")
        model = YOLO(MODEL_PATH).to(device)
        print("✅ Model loaded!")

        # --- 3. Spawn and Get Handles to BOTH QCars ---
        # Instead of possessing, we will spawn them directly here
        print("Spawning and attaching to QCars...")
        car0 = QLabsQCar(qlabs)
        car0.spawn_id(actorNumber=0, location=[-12.82, -4.60, 0], rotation=[0, 0, -0.733], waitForConfirmation=True)
        print("✅ Spawned QCar #0.")

        car1 = QLabsQCar(qlabs)
        car1.spawn_id(actorNumber=1, location=[22.55, 0.81, 0], rotation=[0, 0, 1.57], waitForConfirmation=True)
        print("✅ Spawned QCar #1.")

        time.sleep(1)
        print("Starting detection loops...")

        # --- 4. Main Loop: Get Images from Both Cars ---
        while True:
            # --- Process Car 0 ---
            ok0, image0 = car0.get_image(CAMERA_TO_USE)
            if ok0:
                results0 = model(image0, verbose=False)[0]
                annotated_image0 = results0.plot()
                cv2.imshow("YOLO Detection - Car 0", cv2.cvtColor(annotated_image0, cv2.COLOR_RGB2BGR))

            # --- Process Car 1 ---
            ok1, image1 = car1.get_image(CAMERA_TO_USE)
            if ok1:
                results1 = model(image1, verbose=False)[0]
                annotated_image1 = results1.plot()
                cv2.imshow("YOLO Detection - Car 1", cv2.cvtColor(annotated_image1, cv2.COLOR_RGB2BGR))

            # Check for 'q' key press to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nUser interrupted the script.")
    finally:
        # --- 5. Cleanup ---
        print("Closing connections and cleaning up...")
        if qlabs and qlabs.is_open():
            qlabs.close()
            print("✅ QLabs connection closed.")
        cv2.destroyAllWindows()
        print("✅ Display windows closed.")