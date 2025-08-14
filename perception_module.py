import cv2
import torch
import sys
import time
from ultralytics import YOLO
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2 # IMPORTANT: Using QLabsQCar2 now

# This global flag will be set by the main control scripts to signal a shutdown
KILL_THREAD = False


def run_perception(actor_id):
    """
    This function handles the entire perception pipeline for a single car.
    It's designed to be run in a dedicated thread.

    :param actor_id: The actor number of the QCar to use (e.g., 0 or 1).
    """
    print(f"[Perception-{actor_id}] Starting thread...")

    # Configuration
    MODEL_PATH = "model/best.pt"
    # The CAMERA_RGB constant is the same (4) in both QCar and QCar2 classes
    CAMERA_TO_USE = QLabsQCar2.CAMERA_RGB
    #print(CAMERA_TO_USE)

    qlabs = None
    try:
        # 1. Connect to the QLabs Simulator
        qlabs = QuanserInteractiveLabs()
        if not qlabs.open("localhost"):
            print(
                f"[Perception-{actor_id}] FATAL: Unable to connect to QLabs.",
                file=sys.stderr,
            )
            return
        print(f"[Perception-{actor_id}] ✅ Connection successful!")

        # 2. Load the YOLO Model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLO(MODEL_PATH).to(device)
        print(f"[Perception-{actor_id}] ✅ Model loaded on '{device}'!")

        # 3. Get a handle to the specified QCar using the correct class
        car = QLabsQCar2(qlabs)
        car.actorNumber = actor_id # Manually assign the actor number
        car.possess()
        print(
            f"[Perception-{actor_id}] ✅ Attached to QCar #{actor_id}. Starting video stream..."
        )

        # 4. Warm-up loop to wait for the camera stream
        print(f"[Perception-{actor_id}] Warming up camera stream...")
        start_time = time.time()
        image_ok = False
        while time.time() - start_time < 10:
            image_ok, _ = car.get_image(CAMERA_TO_USE)
            print(image_ok)
            if image_ok:
                print(f"[Perception-{actor_id}] ✅ Camera stream is live!")
                break
            time.sleep(0.5)

        if not image_ok:
            print(f"[Perception-{actor_id}] FATAL: Camera stream failed to start.", file=sys.stderr)
            return

        # 5. Main Detection Loop
        while not KILL_THREAD:
            ok, image = car.get_image(CAMERA_TO_USE)
            if ok:
                results = model(image, verbose=False,conf=0.8)[0]
                annotated_image = results.plot()

                window_name = f"YOLO Detection - Car {actor_id}"
                cv2.imshow(
                    window_name, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.1)

    except Exception as e:
        print(f"[Perception-{actor_id}] An error occurred: {e}", file=sys.stderr)
    finally:
        # 6. Cleanup
        print(f"[Perception-{actor_id}] Shutting down...")
        if qlabs:
            qlabs.close()
        cv2.destroyAllWindows()