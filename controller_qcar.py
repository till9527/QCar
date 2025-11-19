# === controller_qcar.py (Refactored Brain - No QLabs) ===
import multiprocessing
import time
import math
import numpy as np
from multiprocessing.managers import DictProxy
from threading import Thread
from pal.products.qcar import IS_PHYSICAL_QCAR  # Still useful to know
import queue

MAP_SCALER_X = 0.03935
MAP_SCALER_Y = -0.03607  # Note: Y-axis is inverted
MAP_OFFSET_X = 1.3607
MAP_OFFSET_Y = 2.9348
# --- END NEW ---

geofencing_threshold = 1.2
# --- REMOVED: QLabs Imports ---

# --- Perception Thresholds ---
RED_LIGHT_MIN_WIDTH = 30
RED_LIGHT_MIN_HEIGHT = 30
MOVEMENT_THRESHOLD_PX_PER_SEC = 25
STOP_SIGN_MIN_WIDTH = 50
STOP_SIGN_WAIT_TIME_S = 5.0
DANGER_ZONE_LENGTH = 3.0
DANGER_ZONE_WIDTH = 2.0
PIXELS_PER_METER = 60
STALE_OBJECT_TIMEOUT = 1.5
QCAR_DANGER_WIDTH = 140
CAMERA_CENTER_X = 320
CENTER_TOLERANCE = 150
PEDESTRIAN_MIN_WIDTH_FOR_STOP = 90
PEDESTRIAN_MIN_HEIGHT_FOR_STOP = 200
QCAR_MIN_WIDTH_FOR_STOP = 120
SAFE_FOLLOWING_DISTANCE_WIDTH = 80
MIN_DISTANCE_FOR_HARD_BRAKE_WIDTH = 150
DISTANCE_TOLERANCE_WIDTH = 10
LEAD_CAR_CRAWL_SPEED_THRESHOLD = 5.0
ACC_CYCLE_DURATION = 0.5
MAX_SPEED_PXS = 150.0
PEDESTRIAN_CLEAR_TIMEOUT_S = 1.5


# --- V2X Configuration (for Geofencing) ---
# We still need the LOCATIONS of the lights
# ** IMPORTANT: IDs match environment_logic.py (4 and 3) **
# This list is used to map the statuses received from the perception module
# (which should be in the order [ID 4, ID 3]) to the correct geofence area.
traffic_lights = [
    {
        "id": 4,  # Corresponds to ID 4 from perception module
        "location": [23.667, 9.893, 0.005],
        "rotation": [0, 0, 0],
    },
    {
        "id": 3,  # Corresponds to ID 3 from perception module
        "location": [-21.122, 9.341, 0.005],
        "rotation": [0, 0, 180],
    },
]
SCALING_FACTOR = 0.0912
geofencing_threshold = 0.5
# --- REMOVED: qlabs = None ---
MANUAL_GEOFENCE_BOUNDS = {
    4: {  # Bounds for Light ID 4 (Positive X)
        "x_min": 21.125,
        "x_max": 25.0,
        "y_min": -2.274,
        "y_max": 2.158,
    },
    3: {  # Bounds for Light ID 3 (Negative X)
        "x_min": -21.4,
        "x_max": -17.5,
        "y_min": 15.472,
        "y_max": 20.436,
    },
}


# --- V2X Helper Functions (Geofencing only) ---
def generate_geofencing_areas(traffic_lights_list):
    """
    Generates geofencing areas in the car's EKF/Map coordinate system.
    Applies the full affine transformation (scale + offset) for both X and Y.
    """
    generated_areas = []

    # Local offsets for a 0-degree rotation
    local_corner_1 = (1.0, -3.0)
    local_corner_2 = (-3.25, -12.0)

    for light in traffic_lights_list:

        # 1. --- CONVERT LIGHT'S CENTER TO MAP COORDS ---
        raw_world_x = light["location"][0]
        raw_world_y = light["location"][1]

        # --- THIS IS THE CRITICAL CHANGE ---
        # Apply the new, separate transformations
        map_center_x = (raw_world_x * MAP_SCALER_X) + MAP_OFFSET_X
        map_center_y = (raw_world_y * MAP_SCALER_Y) + MAP_OFFSET_Y

        # 2. --- APPLY ROTATION TO LOCAL OFFSETS ---
        try:
            yaw_deg = light["rotation"][2]
        except (KeyError, IndexError):
            yaw_deg = 0.0

        yaw_rad = math.radians(yaw_deg)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)

        # Rotate local_corner_1
        rot_x1 = local_corner_1[0] * cos_yaw - local_corner_1[1] * sin_yaw
        rot_y1 = local_corner_1[0] * sin_yaw + local_corner_1[1] * cos_yaw

        # Rotate local_corner_2
        rot_x2 = local_corner_2[0] * cos_yaw - local_corner_2[1] * sin_yaw
        rot_y2 = local_corner_2[0] * sin_yaw + local_corner_2[1] * cos_yaw

        # 3. --- CREATE FINAL AABB IN MAP COORDS ---
        map_corner_1 = (map_center_x + rot_x1, map_center_y + rot_y1)
        map_corner_2 = (map_center_x + rot_x2, map_center_y + rot_y2)

        x_min = min(map_corner_1[0], map_corner_2[0])
        x_max = max(map_corner_1[0], map_corner_2[0])
        y_min = min(map_corner_1[1], map_corner_2[1])
        y_max = max(map_corner_1[1], map_corner_2[1])

        generated_areas.append(
            {
                "name": f"Traffic Light {light['id']}",
                "bounds": [
                    (x_min, y_min),
                    (x_max, y_max),
                ],
            }
        )
    return generated_areas


def generate_geofencing_areas_visualizer(traffic_lights_list):
    """
    Generates geofencing areas by applying rotation to the
    hard-coded local bounds.

    Assumes `light['rotation']` provides [roll, pitch, yaw].
    Ignores threshold and SCALING_FACTOR.
    """

    generated_areas = []

    # --- Define the "perfect" bounds as local offsets ---
    # (From your example, assuming 0,0,0 rotation)
    local_corner_1 = (1.0, -9.0)  # (local_x1, local_y1)
    local_corner_2 = (-2.0, -13.0)  # (local_x2, local_y2)

    for light in traffic_lights_list:
        # Get the light's world position
        center_x = light["location"][0]
        center_y = light["location"][1]

        # Get the light's Z-axis rotation (yaw) in degrees
        try:
            # Assumes rotation is [roll, pitch, yaw]
            yaw_deg = light["rotation"][2]
        except (KeyError, IndexError):
            # Fallback for safety if 'rotation' isn't provided
            yaw_deg = 0.0

        # Convert to radians for trigonometric functions
        yaw_rad = math.radians(yaw_deg)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)

        # --- Rotate local_corner_1 ---
        # x_rotated = x*cos(a) - y*sin(a)
        # y_rotated = x*sin(a) + y*cos(a)
        rot_x1 = local_corner_1[0] * cos_yaw - local_corner_1[1] * sin_yaw
        rot_y1 = local_corner_1[0] * sin_yaw + local_corner_1[1] * cos_yaw

        # Add rotated offset to center to get final world coordinate
        world_corner_1 = (center_x + rot_x1, center_y + rot_y1)

        # --- Rotate local_corner_2 ---
        rot_x2 = local_corner_2[0] * cos_yaw - local_corner_2[1] * sin_yaw
        rot_y2 = local_corner_2[0] * sin_yaw + local_corner_2[1] * cos_yaw

        # Add rotated offset to center to get final world coordinate
        world_corner_2 = (center_x + rot_x2, center_y + rot_y2)

        # --- Create an Axis-Aligned Bounding Box (AABB) ---
        # Your 'is_inside_geofence' function needs a simple min/max box.
        # We find the min and max x/y values from our two rotated corners.
        x_min = min(world_corner_1[0], world_corner_2[0])
        x_max = max(world_corner_1[0], world_corner_2[0])
        y_min = min(world_corner_1[1], world_corner_2[1])
        y_max = max(world_corner_1[1], world_corner_2[1])

        # Add the final area to the list
        generated_areas.append(
            {
                "name": f"Traffic Light {light['id']}",
                "bounds": [
                    (x_min, y_min),  # (x_min, y_min)
                    (x_max, y_max),  # (x_max, y_max)
                ],
            }
        )

    return generated_areas


# --- REMOVED: get_traffic_lights_status() ---


def is_inside_geofence(position, geofence):
    (x_min, y_min), (x_max, y_max) = geofence
    # print(x_min, y_min, x_max, y_max)
    # print(position)
    return (
        x_min * SCALING_FACTOR <= position[0] <= x_max * SCALING_FACTOR
        and y_min * SCALING_FACTOR <= position[1] <= y_max * SCALING_FACTOR
    )


# --- Helper functions ---
def get_position(results):
    if results and "x" in results[0] and "y" in results[0] and "width" in results[0]:
        x = results[0]["x"]
        width = results[0]["width"]
        height = results[0].get("height", width)
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


# --- MODIFIED: Main function signature (simplified) ---
def main(
    perception_queue: multiprocessing.Queue,
    command_queue: multiprocessing.Queue,
    shared_pose: DictProxy,
):

    # --- V2X State Variables (Local) ---
    is_stopped_v2x_light = False
    geofencing_areas = []
    has_stopped_at = {}

    # --- Perception State Variables ---
    is_stopped_light = False  # For perception-based red light
    is_stopped_pedestrian = False
    tracked_objects = {}
    is_moving_ped = False
    is_stopped_yield_sign = False
    is_stopped_for_sign = False
    is_stopped_qcar_red_light = False
    stop_sign_start_time = 0
    yield_sign_sign_start_time = 0
    last_green_light_seen_time = 0
    red_light_start_time = 0
    is_stopped_qcar = False
    last_stop_qcar = 0
    last_pedestrian_seen_time = 0
    last_stop_seen_time = 0
    last_yield_sign_seen_time = 0
    last_red_light_seen_time = 0
    last_qcar_seen_time = 0
    last_light_seen_time = 0

    # --- Overall Command State ---
    last_command_was_stop = False

    # --- NEW: Timer for printing V2X status ---
    last_v2x_print_time = 0.0

    try:
        # --- Setup Geofencing (No QLabs needed) ---
        if not IS_PHYSICAL_QCAR:
            geofencing_areas = generate_geofencing_areas_visualizer(traffic_lights)
            print(geofencing_areas)
            has_stopped_at = {area["name"]: False for area in geofencing_areas}
            print("[Controller] ✅ Geofencing Initialized (local).")

        # --- *** FIX 1: Initialize persistent variables *before* the loop *** ---
        results = []  # Detections (cleared each loop)
        traffic_light_statuses = []  # V2X Status (persistent)

        # --- Main Controller Loop ---
        while True:
            current_time = time.time()

            # --- 1. GET BUNDLED DATA from Perception Module ---
            results = []  # <-- FIX 1: Clear *only* perception results

            if not perception_queue.empty():
                input_data = perception_queue.get()
                results = input_data.get("detections", [])
                # <-- FIX 1: *Only* update V2X status when new data arrives.
                # It is no longer reset to [] every loop.
                traffic_light_statuses = input_data.get("v2x_statuses", [])

            # --- Print V2X Status periodically ---
            # if (current_time - last_v2x_print_time > 3.0) and traffic_light_statuses:
            #     print(f"[Controller] V2X Status: {traffic_light_statuses}")
            #     last_v2x_print_time = current_time

            # --- 2. RUN V2X/GEOFENCING LOGIC ---
            v2x_wts_to_stop = False  # V2X "wants to stop"

            if not IS_PHYSICAL_QCAR and geofencing_areas:
                position = (shared_pose["x"], shared_pose["y"])

                for i, area in enumerate(geofencing_areas):
                    name = area["name"]
                    inside = is_inside_geofence(position, area["bounds"])
                    try:
                        # Assumes order in traffic_light_statuses matches geofencing_areas
                        traffic_light_status = traffic_light_statuses[i]
                    except IndexError:
                        traffic_light_status = "UNKNOWN"

                        last_light_seen_time = current_time
                        if traffic_light_status == "RED":
                            last_red_light_seen_time = current_time
                            v2x_wts_to_stop = True
                            has_stopped_at[name] = True
                        elif traffic_light_status == "GREEN" and has_stopped_at[name]:
                            last_green_light_seen_time = current_time
                            has_stopped_at[name] = False
                    if not inside and has_stopped_at[name]:
                        has_stopped_at[name] = False

            is_stopped_v2x_light = v2x_wts_to_stop

            # --- 3. PERCEPTION LOGIC (uses 'results' from queue) ---
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

                # Update timestamps
                if cls == "Qcar":
                    last_qcar_seen_time = current_time
                    # print("Qcar width is: ", width)
                    # print("Qcar height is: ", height)
                    # print(
                    #     "These many seconds have passed since we saw a red light: ",
                    #     current_time - last_red_light_seen_time,
                    # )
                if cls == "pedestrian":
                    last_pedestrian_seen_time = current_time
                if cls == "stop_sign":
                    last_stop_seen_time = current_time
                if cls == "yield_sign":
                    last_yield_sign_seen_time = current_time
                if cls == "red_light":
                    last_red_light_seen_time = current_time
                    last_light_seen_time = current_time
                if cls == "green_light":
                    last_light_seen_time = current_time
                    last_green_light_seen_time = current_time
                if cls == "yellow_light":
                    last_light_seen_time = current_time

                # print("last light seen time was: ", last_light_seen_time)
                # print("current time is: ", current_time)
                if position and cls in tracked_objects:
                    last_pos = tracked_objects[cls]["position"]
                    last_time = tracked_objects[cls]["time"]
                    delta_time = current_time - last_time
                    if delta_time > 0:
                        distance = math.hypot(
                            position[0] - last_pos[0], position[1] - last_pos[1]
                        )
                        speed_px_per_sec = distance / delta_time
                        if cls == "pedestrian" and (
                            speed_px_per_sec > MOVEMENT_THRESHOLD_PX_PER_SEC
                        ):
                            is_moving_ped = True
                if position:
                    tracked_objects[cls] = {
                        "position": position,
                        "time": current_time,
                    }

                # --- Perception Stop Conditions ---
                if (
                    cls == "red_light"
                    and width > RED_LIGHT_MIN_WIDTH
                    and height > RED_LIGHT_MIN_HEIGHT
                ):
                    is_stopped_light = True
                    red_light_start_time = current_time
                elif cls == "green_light":
                    is_stopped_light = False

                elif cls == "Qcar" and height > 125:  # Simplified QCar following
                    is_stopped_qcar = True
                    last_stop_qcar = current_time
                elif (
                    cls == "Qcar"
                    and height > 70
                    and width > 70
                    and current_time - last_red_light_seen_time < 5
                ):
                    # print("in front of qcar")
                    is_stopped_qcar_red_light = True

                elif cls == "Qcar" and is_stopped_qcar and height <= 125:
                    is_stopped_qcar = False

                elif (
                    cls == "Qcar"
                    and is_stopped_qcar_red_light
                    and height <= 70
                    and width <= 70
                    and current_time - last_red_light_seen_time > 5
                ):
                    # print("No longer in front of qcar")
                    is_stopped_qcar_red_light = False

                elif (
                    cls == "stop_sign"
                    and not is_stopped_for_sign  # Only trigger once
                    and width > STOP_SIGN_MIN_WIDTH
                    and current_time - stop_sign_start_time > 10  # Cooldown
                ):
                    is_stopped_for_sign = True
                    stop_sign_start_time = current_time

                elif (
                    cls == "yield_sign"
                    and not is_stopped_yield_sign  # Only trigger once
                    and width > 30
                    and current_time - yield_sign_sign_start_time > 6
                ):
                    is_stopped_yield_sign = True
                    yield_sign_sign_start_time = current_time

                elif cls == "pedestrian" and not is_moving_ped:
                    is_stopped_pedestrian = False
                elif (
                    cls == "pedestrian"
                    and is_moving_ped
                    and width > PEDESTRIAN_MIN_WIDTH_FOR_STOP
                    and height > PEDESTRIAN_MIN_HEIGHT_FOR_STOP
                ):
                    is_stopped_pedestrian = True

            # --- 4. TIMEOUT LOGIC ---
            if is_stopped_pedestrian and (
                current_time - last_pedestrian_seen_time > PEDESTRIAN_CLEAR_TIMEOUT_S
            ):
                is_stopped_pedestrian = False

            if is_stopped_qcar and (current_time - last_qcar_seen_time > 1):
                is_stopped_qcar = False

            if is_stopped_qcar_red_light and (
                current_time - last_green_light_seen_time < 1
            ) or (current_time - last_qcar_seen_time > 1):
                is_stopped_qcar_red_light = False

            if is_stopped_for_sign and (
                current_time - stop_sign_start_time > STOP_SIGN_WAIT_TIME_S
            ):
                is_stopped_for_sign = False

            if is_stopped_yield_sign and (
                current_time - yield_sign_sign_start_time > 3
            ):
                is_stopped_yield_sign = False

            if is_stopped_light and (current_time - last_red_light_seen_time > 3):
                is_stopped_light = False

            # --- 5. *** FIX 2: FINAL DECISION BLOCK (with V2X Priority) *** ---

            should_stop = False
            stop_reasons = []

            # Priority 1: V2X Rules (as you requested)
            if is_stopped_v2x_light:
                should_stop = True
                stop_reasons = ["V2X_Light"]

            # Priority 2: Perception Rules
            # Only check these if V2X isn't already stopping us,
            # OR to gather *additional* reasons to stop.
            else:
                all_perception_conditions = {
                    "Perception_Light": is_stopped_light,
                    "Pedestrian": is_stopped_pedestrian,
                    "Stop_Sign": is_stopped_for_sign,
                    "Yield_Sign": is_stopped_yield_sign,
                    "QCar_Too_Close": is_stopped_qcar,
                    "QCar_Too_Close_Light": is_stopped_qcar_red_light,
                }
                # Check if any perception rule wants to stop
                if any(all_perception_conditions.values()):
                    should_stop = True
                    stop_reasons = [
                        k for k, v in all_perception_conditions.items() if v
                    ]

            # Now, send the command based on the final decision
            if should_stop and not last_command_was_stop:
                command_queue.put("STOP")
                last_command_was_stop = True
                print(f"[Controller] STOPPING: Reasons: {stop_reasons}")

            elif not should_stop and last_command_was_stop:
                command_queue.put("GO")
                last_command_was_stop = False
                print("[Controller] RESUMING: All stop conditions clear.")

            # --- 6. LOOP DELAY ---
            time.sleep(0.05)  # Poll V2X and perception at 20Hz

    except KeyboardInterrupt:
        print("[Controller] Shutdown requested.")
    finally:
        # --- REMOVED: QLabs Cleanup ---
        print("[Controller] Process terminated.")
