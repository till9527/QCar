# v2x_module.py (Final Version)

import os

os.environ["IS_PHYSICAL_QCAR"] = "0"
import signal
import numpy as np
from threading import Thread
import time
import sys

# --- Removed Perception/Controller Imports ---
# import multiprocessing as mp
# from perception_module import run_perception
# import perception_module
# import controller_qcar as controller  # The brain

# --- Imports for Spawning (from initCars.py) ---
from qvl.multi_agent import MultiAgent, readRobots
from qvl.real_time import QLabsRealTime

# --- Existing V2X / QCar Imports ---
from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR
from pal.utilities.math import wrap_to_pi
from hal.content.qcar_functions import QCarEKF
from hal.products.mats import SDCSRoadMap
from qvl.traffic_light import QLabsTrafficLight
from qvl.qlabs import QuanserInteractiveLabs


# Define global variable for thread control
global KILL_THREAD
KILL_THREAD = False
global qlabs
qlabs = None


# Used to enable safe keyboard triggered shutdown
def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True
    # perception_module.KILL_THREAD = True # <-- Removed


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
nodeSequence = [10, 2, 4, 6, 13, 19, 17, 20, 22, 10]

# region : Initial setup
if enableSteeringControl:
    roadmap = SDCSRoadMap(leftHandTraffic=False)
    waypointSequence = roadmap.generate_path(nodeSequence)
    initialPose = roadmap.get_node_pose(nodeSequence[0]).squeeze()
else:
    initialPose = [0, 0, 0]

if not IS_PHYSICAL_QCAR:
    calibrate = False
else:
    calibrate = "y" in input("do you want to recalibrate?(y/n)")

calibrationPose = [0, 2, -np.pi / 2]
# endregion

# region : V2X Setup
# Traffic Light Data
traffic_lights = [
    {
        "id": 1,
        "location": [23.667, 9.893, 0.005],
        "rotation": [0, 0, 0],
        "traffic_light_obj": None,
    },
    {
        "id": 2,
        "location": [-21.122, 9.341, 0.005],
        "rotation": [0, 0, 180],
        "traffic_light_obj": None,
    },
]

# Geofencing areas
geofencing_areas = []
SCALING_FACTOR = 0.0912
geofencing_threshold = 1.2
has_stopped_at = {}  # Will be populated in __main__
traffic_light_statuses = ["UNKNOWN"] * len(traffic_lights)
light_threads = []


def generate_geofencing_areas(traffic_lights, threshold):
    return [
        {
            "name": f"Traffic Light {light['id']}",
            "bounds": [
                (
                    light["location"][0] * SCALING_FACTOR - threshold,
                    light["location"][1] * SCALING_FACTOR - threshold,
                ),
                (
                    light["location"][0] * SCALING_FACTOR + threshold,
                    light["location"][1] * SCALING_FACTOR + threshold,
                ),
            ],
        }
        for light in traffic_lights
    ]


def is_inside_geofence(position, geofence):
    (x_min, y_min), (x_max, y_max) = geofence
    return x_min <= position[0] <= x_max and y_min <= position[1] <= y_max


# ========== Traffic Light Sequence Function ==========
def traffic_light_sequence(
    traffic_light, red_time=10, green_time=4, yellow_time=1, delay=0
):
    global KILL_THREAD
    time.sleep(delay)
    while not KILL_THREAD:
        traffic_light.set_color(QLabsTrafficLight.COLOR_RED)
        time.sleep(red_time)
        if KILL_THREAD:
            break
        traffic_light.set_color(QLabsTrafficLight.COLOR_GREEN)
        time.sleep(green_time)
        if KILL_THREAD:
            break
        traffic_light.set_color(QLabsTrafficLight.COLOR_YELLOW)
        time.sleep(yellow_time)


# ========== Traffic Light Status Function ==========
def get_traffic_lights_status():
    global traffic_lights
    try:
        status_map = {0: "NONE", 1: "RED", 2: "YELLOW", 3: "GREEN"}
        statuses = []
        for light in traffic_lights:
            status, color_code = light["traffic_light_obj"].get_color()
            status_str = status_map.get(color_code, "UNKNOWN")
            statuses.append(status_str)
            print(
                f"Traffic Light {light['id']} Status: {status_str}"
            )  # Print added back
        return statuses
    except Exception as e:
        print(f"Error fetching traffic light statuses: {e}")
        return ["UNKNOWN"] * len(traffic_lights)


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


# region : MODIFIED controlLoop (Geofencing logic added back)
# Removed command_queue and shared_pose from arguments
def controlLoop(car_config):
    # region controlLoop setup
    global KILL_THREAD, v_ref, traffic_light_statuses, geofencing_areas, has_stopped_at
    u = 0
    delta = 0

    # This is the *local* speed target for the controller,
    # which will be modified by the geofencing logic.
    # It starts at the global cruise speed (v_ref).
    current_v_ref = v_ref
    # endregion

    # region Controller initialization
    speedController = SpeedController(kp=K_p, ki=K_i)
    if enableSteeringControl:
        steeringController = SteeringController(waypoints=waypointSequence, k=K_stanley)
    # endregion

    # region QCar interface setup
    qcar = QCar(
        readMode=1, frequency=controllerUpdateRate, hilPort=car_config["hilPort"]
    )
    if enableSteeringControl or calibrate:
        ekf = QCarEKF(x_0=initialPose)
        gps = QCarGPS(
            initialPose=calibrationPose,
            calibrate=calibrate,
            gpsPort=car_config["gpsPort"],
            lidarIdealPort=car_config["lidarIdealPort"],
        )
    else:
        gps = memoryview(b"")
    # endregion

    with qcar, gps:
        t0 = time.time()
        t = 0
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

                    # --- GEOFENCING LOGIC (from V2X_simulation.py) ---
                    # Get current position
                    position = (gps.position[0], gps.position[1])

                    for i, area in enumerate(geofencing_areas):
                        name = area["name"]
                        inside = is_inside_geofence(position, area["bounds"])
                        # Read the global status updated by the status thread
                        traffic_light_status = traffic_light_statuses[i]

                        if inside:
                            if (
                                traffic_light_status == "RED"
                                and not has_stopped_at[name]
                            ):
                                current_v_ref = 0.0  # STOP
                                has_stopped_at[name] = True
                                print(f"Stopping at {name} due to RED light!")
                            elif (
                                traffic_light_status == "GREEN" and has_stopped_at[name]
                            ):
                                current_v_ref = v_ref  # GO (resume cruise speed)
                                has_stopped_at[name] = False
                                print(
                                    f"Traffic light at {name} turned GREEN. Resuming movement."
                                )
                        if not inside and has_stopped_at[name]:
                            # Reset stop flag once we leave the area
                            has_stopped_at[name] = False

                    # --- END GEOFENCING LOGIC ---

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

            # --- Removed command_queue check ---

            # region : Update controllers and write to car
            if t < startDelay:
                u = 0
                delta = 0
            else:
                # Use the local current_v_ref, which is controlled by the geofencing logic
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


# region : SIMPLIFIED __main__ (Removed Perception/Controller)
if __name__ == "__main__":
    # --- Removed multiprocessing setup ---
    # mp.set_start_method("spawn", force=True)
    # perception_queue = mp.Queue(maxsize=1)
    # command_queue = mp.Queue(maxsize=1)
    # manager = mp.Manager()
    # shared_pose = manager.dict(...)

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

        # 3. Define and Spawn QCar
        print("Spawning QCar...")
        QCars_to_spawn = [
            {
                "RobotType": "QCar2",
                "Location": [initialPose[0], initialPose[1], 0.0],
                "Rotation": [0, 0, initialPose[2]],
                "Radians": True,
            }
        ]
        MultiAgent(QCars_to_spawn)
        print("✅ QCar spawned and RobotAgents.json created.")

        # 4. Read the config file that was just created
        robotsDir = readRobots()
        Car1 = robotsDir["QC2_0"]

        # 5. Spawn Traffic Lights
        print("Spawning traffic lights...")
        for light in traffic_lights:
            light["traffic_light_obj"] = QLabsTrafficLight(qlabs)
            light["traffic_light_obj"].spawn_id_degrees(
                actorNumber=light["id"],
                location=light["location"],
                rotation=light["rotation"],
                scale=[1, 1, 1],
                waitForConfirmation=False,
            )
        print("✅ Traffic Lights Spawned")

        # 6. Setup Geofencing
        geofencing_areas = generate_geofencing_areas(
            traffic_lights, geofencing_threshold
        )
        has_stopped_at = {area["name"]: False for area in geofencing_areas}
        print("✅ Geofencing Areas Defined")

        # 7. Start traffic light sequence threads
        for i, light in enumerate(traffic_lights):
            t = Thread(
                target=traffic_light_sequence,
                args=(light["traffic_light_obj"], 20, 10, 1, i * 5),
            )
            t.start()
            light_threads.append(t)

        # 8. Start V2X status thread
        statusThread = Thread(target=traffic_light_status_thread)
        statusThread.start()

        # 9. Start All Processes and Threads
        print("Starting Control Loop...")
        # --- Removed Perception and Controller processes ---
        # perception_proc = Thread(...)
        # controller_proc = mp.Process(...)

        # MODIFIED: Pass 'Car1' config directly, no queues
        control_thread = Thread(target=controlLoop, args=(Car1,))
        control_thread.start()
        print("✅ All systems running.")

        # 10. Run main loop
        try:
            while control_thread.is_alive() and (not KILL_THREAD):
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Shutdown initiated by user.")
        finally:
            print("Shutdown initiated...")
            KILL_THREAD = True
            # --- Simplified shutdown ---
            # if controller_proc.is_alive():
            #     controller_proc.terminate()
            # perception_proc.join()
            control_thread.join()
            statusThread.join()
            # controller_proc.join()

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
