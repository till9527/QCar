# === controller.py ===
import multiprocessing
import time

# Threshold
RED_LIGHT_MIN_WIDTH = 38

# Helper functions
def any_detected_objects(results):
    return isinstance(results, list) and len(results) > 0

def get_cls(results):
    return results[0]["class"] if results and "class" in results[0] else None

def get_width(results):
    return results[0]["width"] if results and "width" in results[0] else 0

def main(perception_queue: multiprocessing.Queue, command_queue: multiprocessing.Queue):
    is_stopped = False

    try:
        while True:
            if not perception_queue.empty():
                results = perception_queue.get()

                if any_detected_objects(results):
                    cls = get_cls(results)
                    width = get_width(results)

                    # MODIFIED: Stop condition now only depends on perception.
                    if cls == "Red" and width > RED_LIGHT_MIN_WIDTH and not is_stopped:
                        command_queue.put("STOP")
                        print("[Controller] STOPPING: Red light detected by perception.")
                        is_stopped = True
                        

                    # MODIFIED: Go condition now only depends on perception.
                    elif cls == "Green" and is_stopped:
                        command_queue.put("GO")
                        print("[Controller] RESUMING: Green light detected by perception.")
                        is_stopped = False

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("[Controller] Shutdown requested.")
