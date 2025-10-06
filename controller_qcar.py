# === controller.py ===
import multiprocessing
import time
import math
import numpy as np

# Threshold
RED_LIGHT_MIN_WIDTH = 30
RED_LIGHT_MIN_HEIGHT = 30
MOVEMENT_THRESHOLD_PX_PER_SEC = 25
# --- START: NEW CONSTANTS FOR STOP SIGN ---
STOP_SIGN_MIN_WIDTH = 50  # TUNE: Minimum width in pixels to consider a stop sign valid.
STOP_SIGN_WAIT_TIME_S = 5.0  # Defines the mandatory stop duration in seconds.
# --- END: NEW CONSTANTS FOR STOP SIGN ---
DANGER_ZONE_LENGTH = 3.0  # meters
DANGER_ZONE_WIDTH = 2.0  # meters
PIXELS_PER_METER = 60
# If we haven't seen an object for this long (in seconds), forget about it.
STALE_OBJECT_TIMEOUT = 1.5


# We no longer use these width thresholds for pedestrians/Qcars, but they could be useful for other logic.
# PEDESTRIAN_MIN_WIDTH = 80
# QCAR_MIN_WIDTH = 100
QCAR_DANGER_WIDTH = 140

# TUNE this: Defines the "in front of me" zone.
# Assumes a 640px wide camera image. The center is 320.
CAMERA_CENTER_X = 320
CENTER_TOLERANCE = 150  # How far from the center an object can be (in pixels)

# TUNE this: How close a pedestrian must be before we stop.
PEDESTRIAN_MIN_WIDTH_FOR_STOP = 90
QCAR_MIN_WIDTH_FOR_STOP = 120
SAFE_FOLLOWING_DISTANCE_WIDTH = 80

# TUNE: This is the "emergency brake" distance. If the car gets any closer than
# this, we slam on the brakes. This MUST be larger than the safe following distance.
MIN_DISTANCE_FOR_HARD_BRAKE_WIDTH = 150

# TUNE: This creates a "dead zone" around the safe following distance to prevent
# the car from rapidly accelerating and braking. A larger value is smoother.
DISTANCE_TOLERANCE_WIDTH = 10
# --- Helper functions ---

LEAD_CAR_CRAWL_SPEED_THRESHOLD = 5.0
ACC_CYCLE_DURATION = 0.5  # We will make a decision every half second.

# TUNE: This is the MOST IMPORTANT new value. It's the speed (in px/s) of a
# lead car when it is moving at our car's maximum v_ref. We use this to
# scale the lead car's speed to a 0-1 ratio for our throttle.
MAX_SPEED_PXS = 150.0


def is_object_in_danger_zone(detection, car_heading_rad):
    """
    Estimates an object's position relative to the car and checks if it's
    in a pred_lightefined rectangular "danger zone" in front of the car.
    """
    if not detection:
        return False

    # 1. Get pixel coordinates from detection
    box_width_px = detection.get("width", 0)
    # Get the center of the bottom edge of the bounding box for better distance estimation
    box_center_x_px = detection.get("x", 0) + (box_width_px / 2)

    # 2. Simple distance and lateral deviation estimation
    # Assume the object is on the ground. The further away it is, the smaller it appears.
    # This is a very rough estimate and works best for objects directly ahead.
    if box_width_px < 10:  # Ignore very small (likely far away or false) detections
        return False

    estimated_dist_m = (PIXELS_PER_METER * 1.0) / box_width_px  # Inverse relationship

    # Estimate how far off-center the object is in meters
    # The camera center is 320px
    pixel_offset_from_center = box_center_x_px - CAMERA_CENTER_X
    estimated_lateral_dev_m = pixel_offset_from_center / PIXELS_PER_METER

    # 3. Check if the object is within our rectangular danger zone
    # We are checking a point (estimated_lateral_dev_m, estimated_dist_m)
    # against a box defined by (DANGER_ZONE_WIDTH, DANGER_ZONE_LENGTH)

    is_in_front = 0 < estimated_dist_m < DANGER_ZONE_LENGTH
    is_aligned = abs(estimated_lateral_dev_m) < (DANGER_ZONE_WIDTH / 2)

    if is_in_front and is_aligned:
        # print(f"  [CollisionCheck] DANGER! Object at ~({estimated_lateral_dev_m:.2f}m, {estimated_dist_m:.2f}m)")
        return True

    return False


# Helper functions
def get_position(results):
    if results and "x" in results[0] and "y" in results[0] and "width" in results[0]:
        # Calculate the center of the bounding box for more stable tracking
        x = results[0]["x"]
        width = results[0]["width"]
        # Assuming height is also available or can be inferred_light
        height = results[0].get("height", width)  # Guess height if not present
        center_x = x + width / 2
        center_y = results[0]["y"] + height / 2
        return (center_x, center_y)
    return None


def any_detected_objects(results):
    return isinstance(results, list) and len(results) > 0


def get_cls(results):
    return results[0]["class"] if results and "class" in results[0] else None


def get_width(results):
    return results[0]["width"] if results and "width" in results[0] else 0


def get_x(results):
    return results[0]["x"] if results and "x" in results[0] else 0


def get_y(results):
    return results[0]["y"] if results and "y" in results[0] else 0


def get_height(results):
    return results[0]["height"] if results and "width" in results[0] else 0


def main(perception_queue: multiprocessing.Queue, command_queue: multiprocessing.Queue):
    is_stopped_light = False
    is_stopped_pedestrian = False
    tracked_objects = {}
    is_moving_ped = False
    is_stopped_yield_sign = False
    # --- START: NEW STATE VARIABLES FOR STOP SIGN ---
    is_stopped_for_sign = False
    stop_sign_start_time = 0
    yield_sign_sign_start_time = 0
    stop_sign_width_enough = False
    red_light_start_time = 0
    # --- END: NEW STATE VARIABLES FOR STOP SIGN ---

    ### MODIFICATION 1: Add a variable to track the last time a pedestrian was seen.
    # We initialize it to 0 so that time.time() - last_pedestrian_seen_time is always large at the start.
    last_pedestrian_seen_time = 0
    last_stop_seen_time = 0
    last_yield_sign_seen_time = 0
    last_red_light_seen_time = 0
    # This constant defines the timeout you requested.
    PEDESTRIAN_CLEAR_TIMEOUT_S = 1.5

    try:
        while True:
            # We process the queue if there's data, otherwise we can still check for timeouts.
            if not perception_queue.empty():
                results = perception_queue.get()
                current_time = time.time()
                stale_keys = []
                for key, data in tracked_objects.items():
                    if current_time - data["time"] > STALE_OBJECT_TIMEOUT:
                        stale_keys.append(key)
                for key in stale_keys:
                    del tracked_objects[key]

                if any_detected_objects(results):
                    cls = get_cls(results)
                    width = get_width(results)
                    height = get_height(results)
                    position = get_position(results)
                    is_moving_ped = False

                    ### MODIFICATION 2: If we see a pedestrian, update the timestamp.

                    if cls == "pedestrian":
                        last_pedestrian_seen_time = current_time
                    if cls == "stop_sign":
                        last_stop_seen_time = current_time
                        #print("Stop sign seen with width: ", width)
                    if cls == "yield_sign":
                        last_yield_sign_seen_time = current_time
                    if cls == "red_light":
                        last_red_light_seen_time = current_time
                    if position and cls in tracked_objects:
                        # ==================== LEVEL 2 DEBUGGING ====================
                        # 3. This print tells us we've successfully met the first two conditions.

                        # ==========================================================

                        last_pos = tracked_objects[cls]["position"]
                        last_time = tracked_objects[cls]["time"]
                        delta_time = current_time - last_time

                        if delta_time > 0:
                            distance = math.hypot(
                                position[0] - last_pos[0], position[1] - last_pos[1]
                            )
                            speed_px_per_sec = distance / delta_time

                            # The original debug line
                            if cls == "pedestrian":
                                # print(
                                #     f"      [SPEED CALC] Pedestrian Speed: {speed_px_per_sec:.2f} px/s  |   Threshold is: {MOVEMENT_THRESHOLD_PX_PER_SEC} width is: {width}"
                                # )
                                if speed_px_per_sec > MOVEMENT_THRESHOLD_PX_PER_SEC:
                                    is_moving_ped = True

                        # ==========================================================

                    if position:
                        tracked_objects[cls] = {
                            "position": position,
                            "time": current_time,
                        }
                    # MODIFIED: Stop condition now only depends on perception.
                    if (
                        cls == "red_light"
                        and width > RED_LIGHT_MIN_WIDTH
                        and height>RED_LIGHT_MIN_HEIGHT
                        and not is_stopped_light
                        and not is_stopped_pedestrian
                        and not is_stopped_for_sign
                        and not is_stopped_yield_sign
                    ):
                        command_queue.put("STOP")
                        print(
                            "[Controller] STOPPING: red_light light detected by perception."
                        )
                        is_stopped_light = True
                        red_light_stop_time = time.time()

                    elif (
                        cls == "green_light"
                        and is_stopped_light
                        and not is_stopped_for_sign
                        and not is_stopped_yield_sign
                        and not is_stopped_pedestrian
                    ):
                        command_queue.put("GO")
                        print(
                            "[Controller] RESUMING: green_light light detected by perception."
                        )
                        is_stopped_light = False

                    # --- START: NEW LOGIC FOR STOP SIGN DETECTION ---
                    # This logic triggers the stop for a sign.

                    elif (
                        cls == "stop_sign"  # Assuming perception outputs 'stop_sign'
                        and not is_stopped_for_sign
                        and not is_stopped_light
                        and not is_stopped_pedestrian
                        and not is_stopped_yield_sign
                        and width > STOP_SIGN_MIN_WIDTH
                        and time.time() - stop_sign_start_time > 10
                    ):
                        command_queue.put("STOP")
                        # print("Width of stop sign: ",width)
                        is_stopped_for_sign = True
                        stop_sign_start_time = time.time()  # Start the 5-second timer
                    # --- END: NEW LOGIC FOR STOP SIGN DETECTION ---

                    elif (
                        cls == "yield_sign"
                        and not is_stopped_yield_sign
                        and not is_stopped_for_sign
                        and not is_stopped_light
                        and not is_stopped_pedestrian
                        and width > 55
                        and time.time() - yield_sign_sign_start_time > 6
                    ):
                        command_queue.put("STOP")
                        is_stopped_yield_sign = True
                        yield_sign_sign_start_time = time.time()
                    elif (
                        cls == "pedestrian"
                        and not is_stopped_light
                        and not is_moving_ped
                        and not is_stopped_for_sign
                        and not is_stopped_yield_sign
                        and is_stopped_pedestrian
                    ):
                        command_queue.put("GO")
                        print(
                            ### MODIFICATION 4: Corrected print statement
                            "[Controller] RESUMING: Pedestrian is present but stationary."
                        )
                        is_stopped_pedestrian = False

                    elif (
                        cls == "pedestrian"
                        and not is_stopped_light
                        and is_moving_ped
                        and not is_stopped_pedestrian
                        and not is_stopped_yield_sign
                        and not is_stopped_for_sign
                        and width > PEDESTRIAN_MIN_WIDTH_FOR_STOP
                    ):
                        command_queue.put("STOP")
                        print(
                            ### MODIFICATION 4: Corrected print statement
                            "[Controller] STOPPING: Moving pedestrian detected in path."
                        )
                        is_stopped_pedestrian = True

            ### MODIFICATION 3: Add new logic to resume if a pedestrian has disappeared_light.
            # This check runs outside of the object detection block. It uses the state
            # (`is_stopped_pedestrian`) and the timestamp (`last_pedestrian_seen_time`)
            # to make a decision.
            # if (
            #     not is_stopped_pedestrian
            #     and not is_stopped_light
            #     and not is_stopped_for_sign
            #     and not is_stopped_yield_sign
            #     and (time.time() - last_stop_seen_time == 1)
            #     and not stop_sign_width_enough
            # ):
            #     command_queue.put("STOP")
            #     print(f"[Controller] STOPPING: sign was detected.")
            #     is_stopped_for_sign = True
            #     stop_sign_start_time = time.time()

            if (
                is_stopped_pedestrian
                and not is_stopped_light
                and not is_stopped_for_sign
                and not is_stopped_yield_sign
                and (time.time() - last_pedestrian_seen_time > 1.5)
            ):
                command_queue.put("GO")
                print(
                    f"[Controller] RESUMING: Pedestrian not detected for > {PEDESTRIAN_CLEAR_TIMEOUT_S} second(s)."
                )
                is_stopped_pedestrian = False

            # --- START: NEW LOGIC FOR TIMED RESUME FROM STOP SIGN ---
            # This check runs every cycle. If the car is stopped for a sign and
            # 5 seconds have passed, it will resume, provided no other hazards exist.
            if (
                is_stopped_for_sign
                and (time.time() - stop_sign_start_time > STOP_SIGN_WAIT_TIME_S)
                and not is_stopped_light
                and not is_stopped_pedestrian
                and not is_stopped_yield_sign
            ):
                command_queue.put("GO")
                print(
                    f"[Controller] RESUMING: Stopped at sign for {STOP_SIGN_WAIT_TIME_S} seconds."
                )
                is_stopped_for_sign = False
            # --- END: NEW LOGIC FOR TIMED RESUME FROM STOP SIGN ---
            if (
                is_stopped_yield_sign
                and (time.time() - yield_sign_sign_start_time > 3)
                and not is_stopped_for_sign
                and not is_stopped_light
                and not is_stopped_pedestrian
            ):
                command_queue.put("GO")
                print(f"[Controller] RESUMING: Stopped at yield_sign for 3 seconds.")
                is_stopped_yield_sign = False
            if (
                is_stopped_light
                and (time.time() - red_light_stop_time > 10)
                and not is_stopped_for_sign
                and not is_stopped_yield_sign
                and not is_stopped_pedestrian
            ):
                command_queue.put("GO")
                print(f"[Controller] RESUMING: Stopped at red_light for over 5 seconds.")
                is_stopped_light = False
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("[Controller] Shutdown requested.")
