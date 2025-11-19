# v2x_module.py (Final Version - Modified for 2 Cars)

import os

os.environ["IS_PHYSICAL_QCAR"] = "0"
import signal
import numpy as np
from threading import Thread, Event
import time
import sys
import tkinter as tk
from qvl.qlabs import CommModularContainer

# --- Imports for Spawning (from initCars.py) ---
from qvl.multi_agent import MultiAgent, readRobots
from qvl.real_time import QLabsRealTime
from functools import partial
from qvl.basic_shape import QLabsBasicShape
from qvl.qcar2 import QLabsQCar2

# --- Existing V2X / QCar Imports ---
from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR
from pal.utilities.math import wrap_to_pi
from hal.content.qcar_functions import QCarEKF
from hal.products.mats import SDCSRoadMap
from qvl.traffic_light import QLabsTrafficLight
from qvl.qlabs import QuanserInteractiveLabs
import math

# Define global variable for thread control
global KILL_THREAD
KILL_THREAD = False
global qlabs
qlabs = None
# This will be populated in __main__ to hold stop flags for EACH car
# Format: { "QC2_0": {"Traffic Light 1": False}, "QC2_1": {"Traffic Light 1": False} }
global g_has_stopped_at_by_car
g_has_stopped_at_by_car = {}


# Used to enable safe keyboard triggered shutdown
def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True


signal.signal(signal.SIGINT, sig_handler)

# ================ Experiment Configuration ================
# ===== Timing Parameters
tf = 6000
startDelay = 1
controllerUpdateRate = 100

# ===== Speed Controller Parameters
v_ref = 0.4  # This is the global cruise speed
K_p = 0.1
K_i = 1

# ===== Steering Controller Parameters
enableSteeringControl = True
K_stanley = 1.0

# --- Path and Pose data moved to __main__ ---
# --- so we can define one for each car. ---


# region : V2X Setup (No changes here)
# Traffic Light Data
TRAFFIC_LIGHTS_CONFIG = [
    {"id": 4, "location": [23.667, 9.893, 0.005], "rotation": [0, 0, 0]},
    {"id": 3, "location": [-21.122, 9.341, 0.005], "rotation": [0, 0, 180]},
]


# Geofencing areas
geofencing_areas = []
SCALING_FACTOR = 0.10
geofencing_threshold = 1.2
# has_stopped_at = {} # <-- REPLACED with g_has_stopped_at_by_car
traffic_light_statuses = ["UNKNOWN"] * len(TRAFFIC_LIGHTS_CONFIG)
light_threads = []


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


def is_inside_geofence(position, geofence):
    (x_min, y_min), (x_max, y_max) = geofence
    # print(x_min, y_min, x_max, y_max)
    # print(position)
    return (
        x_min * SCALING_FACTOR <= position[0] <= x_max * SCALING_FACTOR
        and y_min * SCALING_FACTOR <= position[1] <= y_max * SCALING_FACTOR
    )


# ========== Traffic Light Sequence Function ==========
def set_light_color(light_handle, color_const, color_name):
    """Callback function to set a specific light to a specific color."""
    print(f"Setting light {light_handle.actorNumber} to {color_name}")
    light_handle.set_color(color_const)


def traffic_light_sequence(
    traffic_light, stop_event, red_time=15, green_time=30, yellow_time=1, delay=0
):
    """
    Controls the R-Y-G sequence for a single traffic light in a continuous loop.
    Checks a stop_event to allow for graceful termination.
    """
    try:
        # time.sleep(delay)
        # Use wait() instead of sleep() so it can be interrupted
        if stop_event.wait(delay):
            print(
                f"Light {traffic_light.actorNumber} sequence stopped during initial delay."
            )
            return  # Stop requested before we even started

        while not stop_event.is_set():
            # Check flag before each action
            if stop_event.is_set():
                break
            traffic_light.set_color(QLabsTrafficLight.COLOR_GREEN)
            # wait() returns True if event was set, False if it timed out
            if stop_event.wait(green_time):
                break  # Stop requested

            if stop_event.is_set():
                break
            traffic_light.set_color(QLabsTrafficLight.COLOR_YELLOW)
            if stop_event.wait(yellow_time):
                break  # Stop requested

            if stop_event.is_set():
                break
            traffic_light.set_color(QLabsTrafficLight.COLOR_RED)
            if stop_event.wait(red_time):
                break  # Stop requested

        print(f"Traffic light {traffic_light.actorNumber} sequence stopped.")

    except Exception as e:
        print(f"Error in traffic_light_sequence for {traffic_light.actorNumber}: {e}")
        # This can happen if the QLabs connection is closed while the thread is waiting
        pass


class TrafficControlApp:
    def __init__(self, root, light_handles):
        self.root = root
        self.light_4_handle = light_handles[0]
        self.light_3_handle = light_handles[1]

        self.light_threads = []
        self.stop_events = []

        # State variable for the toggle
        self.auto_mode_var = tk.BooleanVar(value=True)

        self.setup_ui()
        self.on_toggle()  # Call once to set initial state (auto mode)

    def setup_ui(self):
        # --- Frame for Toggle ---
        toggle_frame = tk.Frame(self.root)
        toggle_frame.pack(fill=tk.X, padx=10, pady=5)

        self.toggle_button = tk.Checkbutton(
            toggle_frame,
            text="Automatic Mode",
            variable=self.auto_mode_var,
            command=self.on_toggle,
            font=("Helvetica", 10, "bold"),
        )
        self.toggle_button.pack()

        # --- Frame for Light 4 ---
        frame4 = tk.Frame(self.root, relief=tk.RIDGE, borderwidth=2)
        frame4.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(frame4, text="Light ID 4 (Positive X)").pack()

        self.btn_4_g = tk.Button(
            frame4,
            text="Green",
            bg="#70e070",
            command=partial(
                set_light_color,
                self.light_4_handle,
                QLabsTrafficLight.COLOR_GREEN,
                "GREEN",
            ),
        )
        self.btn_4_g.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.btn_4_y = tk.Button(
            frame4,
            text="Yellow",
            bg="#f0e070",
            command=partial(
                set_light_color,
                self.light_4_handle,
                QLabsTrafficLight.COLOR_YELLOW,
                "YELLOW",
            ),
        )
        self.btn_4_y.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.btn_4_r = tk.Button(
            frame4,
            text="Red",
            bg="#f07070",
            command=partial(
                set_light_color, self.light_4_handle, QLabsTrafficLight.COLOR_RED, "RED"
            ),
        )
        self.btn_4_r.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # --- Frame for Light 3 ---
        frame3 = tk.Frame(self.root, relief=tk.RIDGE, borderwidth=2)
        frame3.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(frame3, text="Light ID 3 (Negative X)").pack()

        self.btn_3_g = tk.Button(
            frame3,
            text="Green",
            bg="#70e070",
            command=partial(
                set_light_color,
                self.light_3_handle,
                QLabsTrafficLight.COLOR_GREEN,
                "GREEN",
            ),
        )
        self.btn_3_g.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.btn_3_y = tk.Button(
            frame3,
            text="Yellow",
            bg="#f0e070",
            command=partial(
                set_light_color,
                self.light_3_handle,
                QLabsTrafficLight.COLOR_YELLOW,
                "YELLOW",
            ),
        )
        self.btn_3_y.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.btn_3_r = tk.Button(
            frame3,
            text="Red",
            bg="#f07070",
            command=partial(
                set_light_color, self.light_3_handle, QLabsTrafficLight.COLOR_RED, "RED"
            ),
        )
        self.btn_3_r.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # Store buttons for easy access
        self.manual_buttons = [
            self.btn_4_g,
            self.btn_4_y,
            self.btn_4_r,
            self.btn_3_g,
            self.btn_3_y,
            self.btn_3_r,
        ]

    def set_manual_buttons_state(self, state):
        """Helper function to enable/disable all manual buttons."""
        for btn in self.manual_buttons:
            btn.config(state=state)

    def on_toggle(self):
        """Called when the toggle button is clicked."""
        if self.auto_mode_var.get():
            # --- SWITCHING TO AUTOMATIC ---
            print("Switching to AUTOMATIC mode.")
            self.set_manual_buttons_state(tk.DISABLED)
            self.start_auto_sequences()
        else:
            # --- SWITCHING TO MANUAL ---
            print("Switching to MANUAL mode.")
            self.stop_auto_sequences()
            self.set_manual_buttons_state(tk.NORMAL)

    def start_auto_sequences(self):
        """Stops any existing sequences and starts new ones."""
        self.stop_auto_sequences()  # Ensure old ones are stopped

        print("Starting automatic light sequences...")

        # Create event and thread for Light 4
        stop_event_4 = Event()
        t4 = Thread(
            target=traffic_light_sequence,
            args=(self.light_4_handle, stop_event_4, 15, 30, 1, 0),  # 0s delay
            daemon=True,  # Make daemon so it exits if main thread dies
        )

        # Create event and thread for Light 3
        stop_event_3 = Event()
        t3 = Thread(
            target=traffic_light_sequence,
            args=(
                self.light_3_handle,
                stop_event_3,
                15,
                30,
                1,
                16,
            ),  # 16s delay (Yellow + Red)
            daemon=True,
        )

        self.stop_events = [stop_event_4, stop_event_3]
        self.light_threads = [t4, t3]

        t4.start()
        t3.start()

    def stop_auto_sequences(self):
        """Signals all running traffic light threads to stop."""
        if not self.light_threads:
            return  # Nothing to stop

        print("Stopping automatic light sequences...")

        for event in self.stop_events:
            event.set()  # Signal threads to stop

        for thread in self.light_threads:
            thread.join(timeout=0.5)  # Wait briefly for them to exit

        self.light_threads = []
        self.stop_events = []


# ========== Traffic Light Status Function ==========
def get_traffic_lights_status():
    global TRAFFIC_LIGHTS_CONFIG
    try:
        status_map = {0: "NONE", 1: "RED", 2: "YELLOW", 3: "GREEN"}
        statuses = []
        for light in TRAFFIC_LIGHTS_CONFIG:
            status, color_code = light["traffic_light_obj"].get_color()
            status_str = status_map.get(color_code, "UNKNOWN")
            statuses.append(status_str)
            # Print removed to avoid clutter with two cars
            # print(f"Traffic Light {light['id']} Status: {status_str}")
        return statuses
    except Exception as e:
        print(f"Error fetching traffic light statuses: {e}")
        return ["UNKNOWN"] * len(TRAFFIC_LIGHTS_CONFIG)


def traffic_light_status_thread():
    global traffic_light_statuses
    while not KILL_THREAD:
        traffic_light_statuses = get_traffic_lights_status()
        time.sleep(1)


# endregion


# region : Controller Classes (no changes)
class SpeedController:
    def __init__(self, kp=0, ki=0):
        self.maxThrottle = 0.3
        self.kp = kp
        self.ki = ki
        self.ei = 0

    def update(self, v, v_ref, dt):
        e = v_ref - v
        self.ei += dt * e
        return np.clip(
            self.kp * e + self.ki * self.ei, -self.maxThrottle, self.maxThrottle
        )


class SteeringController:
    def __init__(self, waypoints, k=1, cyclic=True):
        self.maxSteeringAngle = np.pi / 6
        self.wp = waypoints
        self.N = len(waypoints[0, :])
        self.wpi = 0
        self.k = k
        self.cyclic = cyclic
        self.p_ref = (0, 0)
        self.th_ref = 0

    def update(self, p, th, speed):
        wp_1 = self.wp[:, np.mod(self.wpi, self.N - 1)]
        wp_2 = self.wp[:, np.mod(self.wpi + 1, self.N - 1)]
        v = wp_2 - wp_1
        v_mag = np.linalg.norm(v)
        try:
            v_uv = v / v_mag
        except ZeroDivisionError:
            return 0
        tangent = np.arctan2(v_uv[1], v_uv[0])
        s = np.dot(p - wp_1, v_uv)
        if s >= v_mag:
            if self.cyclic or self.wpi < self.N - 2:
                self.wpi += 1
        ep = wp_1 + v_uv * s
        ct = ep - p
        dir = wrap_to_pi(np.arctan2(ct[1], ct[0]) - tangent)
        ect = np.linalg.norm(ct) * np.sign(dir)
        psi = wrap_to_pi(tangent - th)
        self.p_ref = ep
        self.th_ref = tangent
        return np.clip(
            wrap_to_pi(psi + np.arctan2(self.k * ect, speed)),
            -self.maxSteeringAngle,
            self.maxSteeringAngle,
        )


# endregion


# region : MODIFIED controlLoop (Takes car-specific path and state)
def controlLoop(
    car_config,
    car_id,
    car_initial_pose,
    car_waypoint_sequence,
    car_calibration_pose,
    calibrate_car,
):
    # region controlLoop setup
    # Make sure to access the *global* variables
    global KILL_THREAD, v_ref, traffic_light_statuses, geofencing_areas, g_has_stopped_at_by_car
    u = 0
    delta = 0

    # Get this car's specific dictionary for stop flags
    my_stop_flags = g_has_stopped_at_by_car[car_id]

    current_v_ref = v_ref
    # endregion

    # region Controller initialization
    speedController = SpeedController(kp=K_p, ki=K_i)
    if enableSteeringControl:
        # Use the car-specific waypoint sequence
        steeringController = SteeringController(
            waypoints=car_waypoint_sequence, k=K_stanley
        )
    # endregion

    # region QCar interface setup
    qcar = QCar(
        readMode=1, frequency=controllerUpdateRate, hilPort=car_config["hilPort"]
    )
    if enableSteeringControl or calibrate_car:
        # Use the car-specific initial pose for the EKF
        ekf = QCarEKF(x_0=car_initial_pose)
        gps = QCarGPS(
            initialPose=car_calibration_pose,  # Use shared calibration pose
            calibrate=calibrate_car,
            gpsPort=car_config["gpsPort"],
            lidarIdealPort=car_config["lidarIdealPort"],
        )
    else:
        gps = memoryview(b"")
    # endregion

    with qcar, gps:
        t0 = time.time()
        t = 0
        qlabs_car = QLabsQCar2(qlabs)
        qlabs_car.actorNumber = car_config["actorNumber"]
        while (t < tf + startDelay) and (not KILL_THREAD):
            # region : Loop timing update
            tp = t
            t = time.time() - t0
            dt = t - tp
            # endregion

            # region : Read from sensors and update state estimates
            qcar.read()
            if enableSteeringControl:
                if gps.readGPS():

                    # --- GEOFENCING LOGIC (Modified) ---
                    position = (gps.position[0], gps.position[1])
                    is_inside_any_geofence = False

                    for i, area in enumerate(geofencing_areas):
                        name = area["name"]
                        inside = is_inside_geofence(position, area["bounds"])
                        traffic_light_status = traffic_light_statuses[i]

                        if inside:
                            # Use this car's specific stop flag
                            is_inside_any_geofence = True
                            # print("is inside geofence")
                            if (
                                traffic_light_status == "RED"
                                and not my_stop_flags[name]
                            ):
                                current_v_ref = 0.0  # STOP
                                my_stop_flags[name] = True
                                print(
                                    f"[{car_id}] Stopping at {name} due to RED light!"
                                )
                            elif (
                                traffic_light_status == "GREEN" and my_stop_flags[name]
                            ):
                                current_v_ref = v_ref  # GO
                                my_stop_flags[name] = False
                                print(
                                    f"[{car_id}] Traffic light at {name} turned GREEN. Resuming."
                                )
                        if not inside:
                            qlabs_car.set_led_strip_uniform(
                                [0.0, 1.0, 0.0], waitForConfirmation=False
                            )
                        if not inside and my_stop_flags[name]:
                            my_stop_flags[name] = False
                    # --- END GEOFENCING LOGIC ---
                    if is_inside_any_geofence:
                        # Red if inside ANY geofence
                        qlabs_car.set_led_strip_uniform(
                            [1.0, 0.0, 0.0], waitForConfirmation=False
                        )
                    else:
                        # Green (or White) if outside ALL geofences
                        qlabs_car.set_led_strip_uniform(
                            [0.0, 1.0, 0.0], waitForConfirmation=False
                        )
                    y_gps = np.array(
                        [gps.position[0], gps.position[1], gps.orientation[2]]
                    )
                    ekf.update(
                        [qcar.motorTach, delta],
                        dt,
                        y_gps,
                        qcar.gyroscope[2],
                    )
                else:
                    ekf.update(
                        [qcar.motorTach, delta],
                        dt,
                        None,
                        qcar.gyroscope[2],
                    )

                x = ekf.x_hat[0, 0]
                y = ekf.x_hat[1, 0]
                th = ekf.x_hat[2, 0]
                p = np.array([x, y]) + np.array([np.cos(th), np.sin(th)]) * 0.2
            v = qcar.motorTach
            # endregion

            # region : Update controllers and write to car
            if t < startDelay:
                u = 0
                delta = 0
            else:
                u = speedController.update(v, current_v_ref, dt)
                if enableSteeringControl:
                    delta = steeringController.update(p, th, v)
                else:
                    delta = 0

            qcar.write(u, delta)
            # endregion
            continue
        qcar.read_write_std(throttle=0, steering=0)


# endregion


# region : HEAVILY MODIFIED __main__ (Spawns 2 cars)
if __name__ == "__main__":

    # --- Setup that was previously global ---
    spawned_visualizers = []
    if enableSteeringControl:
        roadmap = SDCSRoadMap(leftHandTraffic=False)

        # Path for Car 1 (Outer Loop)
        nodeSequence1 = [10, 2, 4, 6, 13, 19, 17, 20, 22, 10]
        waypointSequence1 = roadmap.generate_path(nodeSequence1)
        initialPose1 = roadmap.get_node_pose(nodeSequence1[0]).squeeze()

        # Path for Car 2 (Inner Loop)
        nodeSequence2 = [1, 7, 5, 3, 1]
        waypointSequence2 = roadmap.generate_path(nodeSequence2)
        initialPose2 = roadmap.get_node_pose(nodeSequence2[0]).squeeze()
    else:
        initialPose1 = [0, 0, 0]
        initialPose2 = [2, 0, 0]  # Just give it a different start pos

    if not IS_PHYSICAL_QCAR:
        calibrate = False
    else:
        calibrate = "y" in input("do you want to recalibrate?(y/n)")

    # This can be shared by both
    calibrationPose = [0, 2, -np.pi / 2]
    # --- End of setup ---

    # --- Main QLabs & Spawning Block ---
    try:
        # 1. Connect to QLabs
        print("Connecting to QLabs...")
        qlabs = QuanserInteractiveLabs()
        qlabs.open("localhost")
        print("✅ Connected to QLabs.")

        # 2. Clean up environment
        print("Cleaning up existing actors and models...")
        QLabsRealTime().terminate_all_real_time_models()
        time.sleep(1)
        qlabs.destroy_all_spawned_actors()
        print("✅ Environment is clean.")

        # 3. Define and Spawn 2 QCars
        print("Spawning 2 QCars...")
        QCars_to_spawn = [
            {
                "RobotType": "QCar2",
                "Location": [initialPose1[0], initialPose1[1], 0.0],
                "Rotation": [0, 0, initialPose1[2]],
                "Radians": True,
            },
            {
                "RobotType": "QCar2",
                "Location": [initialPose2[0], initialPose2[1], 0.0],
                "Rotation": [0, 0, initialPose2[2]],
                "Radians": True,
            },
        ]
        MultiAgent(QCars_to_spawn)
        print("✅ 2 QCars spawned and RobotAgents.json created.")

        # 4. Read the config file
        robotsDir = readRobots()
        # Get configs and IDs for both cars
        Car1_ID = "QC2_0"
        Car2_ID = "QC2_1"
        Car1_Config = robotsDir[Car1_ID]
        Car2_Config = robotsDir[Car2_ID]

        # 5. Spawn Traffic Lights
        traffic_light_handles = []
        for config in TRAFFIC_LIGHTS_CONFIG:
            light = QLabsTrafficLight(qlabs)
            light.spawn_id_degrees(
                actorNumber=config["id"],
                location=config["location"],
                rotation=config["rotation"],
                scale=[1, 1, 1],
                configuration=0,
                waitForConfirmation=True,
            )
            traffic_light_handles.append(light)
            config["traffic_light_obj"] = light
        print(f"Spawned {len(traffic_light_handles)} traffic lights.")
        print("Calculating geofence areas for visualization...")
        geofence_areas_viz = generate_geofencing_areas_visualizer(
            TRAFFIC_LIGHTS_CONFIG  # Use the imported constant
        )
        print(f"Found {len(geofence_areas_viz)} areas. Spawning visualizers...")

        # Use a high actor ID to avoid collisions
        actor_id_counter = 500

        for i, area in enumerate(geofence_areas_viz):
            bounds = area["bounds"]
            x_min = bounds[0][0]
            y_min = bounds[0][1]
            x_max = bounds[1][0]
            y_max = bounds[1][1]

            # Use the name for printing
            light_name = area["name"]

            # Calculate spawn parameters from the bounds
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            size_x = x_max - x_min
            size_y = y_max - y_min

            actor_id = 150 + i

            print(f"\nProcessing {light_name}:")
            print(
                f"  ... Spawning box at [{center_x:.2f}, {center_y:.2f}] (ID: {actor_id})"
            )
            print(f"  ... with size [X={size_x:.2f}, Y={size_y:.2f}]")

            shape = QLabsBasicShape(qlabs)

            # The rest of the QLabs spawning logic remains the same
            shape.spawn_id(
                actorNumber=int(actor_id),
                location=[float(center_x), float(center_y), 0.01],
                rotation=[0.0, 0.0, 0.0],
                scale=[float(size_x), float(size_y), 0.01],
                configuration=int(shape.SHAPE_CUBE),
                waitForConfirmation=True,
            )
            try:
                shape.set_material_properties(
                    color=[1.0, 0.0, 0.0, 0.4],
                    metallic=0.1,
                    roughness=0.8,
                    waitForConfirmation=True,
                )
                print("  ... Set material to RED (transparent).")
            except Exception as e:
                print(f"  ... Could not set transparency, setting to OPAQUE RED.")
                try:
                    shape.set_material_properties(
                        color=[1.0, 0.0, 0.0],
                        metallic=0.1,
                        roughness=0.8,
                        waitForConfirmation=True,
                    )
                except Exception as e2:
                    print(f"  ... Could not set color (using default): {e2}")

        # 6. Setup Geofencing

        geofencing_areas = generate_geofencing_areas_visualizer(TRAFFIC_LIGHTS_CONFIG)
        # Create a sub-dictionary for each car
        area_names = [area["name"] for area in geofencing_areas]
        g_has_stopped_at_by_car[Car1_ID] = {name: False for name in area_names}
        g_has_stopped_at_by_car[Car2_ID] = {name: False for name in area_names}
        print("✅ Geofencing Areas Defined (for 2 cars)")

        # 7. Start traffic light sequence threads

        # 8. Start V2X status thread (only need one)
        statusThread = Thread(target=traffic_light_status_thread)
        statusThread.start()

        # 9. Start All Processes and Threads
        print("Starting Control Loops for 2 Cars...")

        # Create and start a thread for Car 1
        control_thread_1 = Thread(
            target=controlLoop,
            args=(
                Car1_Config,
                Car1_ID,
                initialPose1,
                waypointSequence1,
                calibrationPose,
                calibrate,
            ),
        )

        # Create and start a thread for Car 2
        control_thread_2 = Thread(
            target=controlLoop,
            args=(
                Car2_Config,
                Car2_ID,
                initialPose2,
                waypointSequence2,
                calibrationPose,
                calibrate,  # Note: Both cars get the same calibration setting
            ),
        )

        control_thread_1.start()
        control_thread_2.start()
        root = tk.Tk()
        root.title("Traffic Light Control")

        # Get the handles we spawned earlier
        light_4_handle = traffic_light_handles[0]
        light_3_handle = traffic_light_handles[1]

        # Instantiate the app
        app = TrafficControlApp(root, [light_4_handle, light_3_handle])

        # Disable the window's close button
        root.protocol(
            "WM_DELETE_WINDOW",
            lambda: print("UI closing is disabled. Use Ctrl+C in console to stop."),
        )

        print("UI is running. Press Ctrl+C in the console to quit.")
        root.mainloop()
        print("✅ All systems running.")

        # 10. Run main loop (wait for BOTH threads to finish)
        try:
            while (control_thread_1.is_alive() or control_thread_2.is_alive()) and (
                not KILL_THREAD
            ):
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Shutdown initiated by user.")
        finally:
            print("Shutdown initiated...")
            KILL_THREAD = True

            # Wait for both control threads to join
            control_thread_1.join()
            control_thread_2.join()
            statusThread.join()
            print("✅ Control and status threads joined.")

    except Exception as e:
        print(f"An error occurred during setup or execution: {e}")
        KILL_THREAD = True

    finally:
        # 11. Final Cleanup
        print("Stopping traffic light threads...")
        for t in light_threads:
            t.join()

        if qlabs:
            print("Closing QLabs connection...")
            qlabs.close()
            print("✅ QLabs Closed.")

        if not IS_PHYSICAL_QCAR:
            QLabsRealTime().terminate_all_real_time_models()

        print("Experiment complete.")
# endregion
