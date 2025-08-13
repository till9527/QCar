import cv2
import torch
import sys
import time
from ultralytics import YOLO
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar import QLabsQCar

# --- Configuration ---
MODEL_PATH = "model/best.pt"
CAMERA_TO_USE = QLabsQCar.CAMERA_RGB

# --- Main Execution ---
if __name__ == '__main__':
    qlabs = None
    try:
        # --- 1. Connect to the QLabs Simulator ---
        print("Connecting to QLabs...")
        qlabs = QuanserInteractiveLabs()
        if not qlabs.open("localhost"):
            print("FATAL: Unable to connect to QLabs. Is the simulation running?")
            sys.exit(1)
        print("✅ Connection successful!")

        # --- 2. Load the YOLO Model ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading YOLO model on device: '{device}'...")
        model = YOLO(MODEL_PATH).to(device)
        print("✅ Model loaded!")

        # --- 3. Get a Handle to the QCar ---
        car = QLabsQCar(qlabs)
        car.spawn_id(
            actorNumber=0,
            location=[0, 0, 0],
            rotation=[0, 0, 0],
            waitForConfirmation=False
        )
        print("✅ Attached to QCar with Actor Number 0. Starting video stream...")
        time.sleep(1)

        # --- 4. Main Loop: Get Image -> Run Detection -> Display ---
        while True:
            ok, image = car.get_image(CAMERA_TO_USE)
            if ok:
                results = model(image, verbose=False)[0]
                annotated_image = results.plot()

                # ✅ FIX 1: Corrected the typo to COLOR_RGB2BGR
                cv2.imshow("YOLOv8 Live Detection", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Failed to get image. Retrying...")
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nUser interrupted the script.")
    finally:
        # --- 5. Cleanup ---
        print("Closing connections and cleaning up...")

        # ✅ FIX 2: Removed the .is_open() check which doesn't exist
        if qlabs:
            qlabs.close()
            print("✅ QLabs connection closed.")

        cv2.destroyAllWindows()
        print("✅ Display windows closed.")