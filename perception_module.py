# perception_module.py (Modified)
import cv2
import torch
import sys
import time
from ultralytics import YOLO
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2

KILL_THREAD = False


def run_perception(perception_queue, actor_id):
    """
    This function handles the perception pipeline and sends results to a queue.
    """
    print(f"[Perception-{actor_id}] Starting thread...")
    MODEL_PATH = "model/best.pt"
    CAMERA_TO_USE = QLabsQCar2.CAMERA_RGB

    qlabs = None
    try:
        qlabs = QuanserInteractiveLabs()
        qlabs.open("localhost")
        print(f"[Perception-{actor_id}] ✅ Connection successful!")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLO(MODEL_PATH).to(device)
        print(f"[Perception-{actor_id}] ✅ Model loaded on '{device}'!")

        car = QLabsQCar2(qlabs)
        car.actorNumber = actor_id
        car.possess()
        print(f"[Perception-{actor_id}] ✅ Attached to QCar #{actor_id}.")

        # Main Detection Loop
        while not KILL_THREAD:
            ok, image = car.get_image(CAMERA_TO_USE)
            if ok:
                # Run inference
                results = model(image, verbose=False, conf=0.8)[0]

                # --- NEW: Process and send detection data ---
                detections = []
                for box in results.boxes:
                    class_id = int(box.cls)
                    class_name = model.names[class_id]
                    x_center, y_center, width, height = box.xywh[0]
                    x_top_left = x_center.item() - (width.item() / 2)
                    y_top_left = y_center.item() - (height.item() / 2)

                    detection_data = {
                        "class": class_name,
                        "width": width.item(),
                        "height": height.item(),
                        "x": x_top_left,
                        "y": y_top_left,
                    }
                    detections.append(detection_data)

                # Put the list of detections into the queue for the controller
                if not perception_queue.full():
                    perception_queue.put(detections)
                # --- END NEW ---

                # Optional: still show the annotated image
                annotated_image = results.plot()
                window_name = f"YOLO Detection - Car {actor_id}"
                cv2.imshow(window_name, annotated_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.01)

    except Exception as e:
        print(f"[Perception-{actor_id}] An error occurred: {e}", file=sys.stderr)
    finally:
        print(f"[Perception-{actor_id}] Shutting down...")
        if qlabs:
            qlabs.close()
        cv2.destroyAllWindows()
