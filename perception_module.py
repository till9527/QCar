# perception_module.py (Modified)
import cv2
import torch
import sys
import time
from ultralytics import YOLO
from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2
from pal.products.qcar import IS_PHYSICAL_QCAR

# --- NEW: Import for V2X ---
from qvl.traffic_light import QLabsTrafficLight

KILL_THREAD = False

# --- NEW: V2X Configuration ---
# These are the lights spawned by environment_logic.py
TRAFFIC_LIGHTS_CONFIG = [
    {"id": 4},
    {"id": 3},
]
# We will populate this list of handles
traffic_light_handles = []

# --- NEW: V2X Helper Function ---
def get_traffic_lights_status():
    global traffic_light_handles
    try:
        status_map = {0: "NONE", 1: "RED", 2: "YELLOW", 3: "GREEN"}
        statuses = []
        for light in traffic_light_handles:
            status, color_code = light.get_color()
            status_str = status_map.get(color_code, "UNKNOWN")
            statuses.append(status_str)
        return statuses
    except Exception as e:
        # Don't print an error every loop, just return UNKNOWN
        return ["UNKNOWN"] * len(traffic_light_handles)


def run_perception(perception_queue, actor_id):
    """
    This function handles the perception pipeline AND V2X data gathering.
    It sends results to a queue.
    """
    print(f"[Perception-{actor_id}] Starting thread...")
    MODEL_PATH = "model/best.pt"
    CAMERA_TO_USE = QLabsQCar2.CAMERA_RGB

    qlabs = None
    try:
        qlabs = QuanserInteractiveLabs()
        qlabs.open("localhost")
        print(f"[Perception-{actor_id}] ✅ Connection successful!")
        print("Is Cuda available?",torch.cuda.is_available())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLO(MODEL_PATH).to(device)
        print(f"[Perception-{actor_id}] ✅ Model loaded on '{device}'!")

        car = QLabsQCar2(qlabs)
        car.actorNumber = actor_id
        car.possess()
        print(f"[Perception-{actor_id}] ✅ Attached to QCar #{actor_id}.")

        # --- NEW: V2X Setup ---
        global traffic_light_handles
        if not IS_PHYSICAL_QCAR:
            for config in TRAFFIC_LIGHTS_CONFIG:
                light = QLabsTrafficLight(qlabs)
                # We don't spawn, just get a handle to the existing light
                light.actorNumber = config["id"]
                traffic_light_handles.append(light)
            print(f"[Perception-{actor_id}] ✅ Attached to {len(traffic_light_handles)} traffic lights.")
        # --- END NEW ---

        # Main Detection Loop
        while not KILL_THREAD:
            ok, image = car.get_image(CAMERA_TO_USE)
            if ok:
                results = model(image, device=device, verbose=False)[0]

                # --- NEW: Bundle perception AND v2x data ---
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

                # --- NEW: Get V2X Statuses ---
                v2x_statuses = []
                if not IS_PHYSICAL_QCAR:
                    v2x_statuses = get_traffic_lights_status()
                
                # --- NEW: Send bundled data dictionary ---
                output_data = {
                    "detections": detections,
                    "v2x_statuses": v2x_statuses
                }
                
                if not perception_queue.full():
                    perception_queue.put(output_data)
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
        print(f"[Perception-{actor_id}] Stopping thread.")
        if qlabs:
            qlabs.close()
        cv2.destroyAllWindows()
