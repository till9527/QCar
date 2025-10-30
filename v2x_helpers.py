# v2x_module.py (Final Version - Modified for 2 Cars)

import os

os.environ["IS_PHYSICAL_QCAR"] = "0"
import signal
import numpy as np
from threading import Thread
import time
import sys

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
# has_stopped_at = {} # <-- REPLACED with g_has_stopped_at_by_car
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
            # Print removed to avoid clutter with two cars
            # print(f"Traffic Light {light['id']} Status: {status_str}")
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

                    for i, area in enumerate(geofencing_areas):
                        name = area["name"]
                        inside = is_inside_geofence(position, area["bounds"])
                        traffic_light_status = traffic_light_statuses[i]

                        if inside:
                            # Use this car's specific stop flag
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
                        if not inside and my_stop_flags[name]:
                            my_stop_flags[name] = False
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
        # Create a sub-dictionary for each car
        area_names = [area["name"] for area in geofencing_areas]
        g_has_stopped_at_by_car[Car1_ID] = {name: False for name in area_names}
        g_has_stopped_at_by_car[Car2_ID] = {name: False for name in area_names}
        print("✅ Geofencing Areas Defined (for 2 cars)")

        # 7. Start traffic light sequence threads
        for i, light in enumerate(traffic_lights):
            t = Thread(
                target=traffic_light_sequence,
                args=(light["traffic_light_obj"], 20, 10, 1, i * 5),
            )
            t.start()
            light_threads.append(t)

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
