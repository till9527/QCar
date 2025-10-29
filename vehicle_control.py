# vehicle_control.py (Modified)

# region : Imports
import os
import signal
import numpy as np
import time
import multiprocessing as mp
from threading import Thread

# --- NEW: Import the new modules ---
from perception_module import run_perception
import perception_module
import controller_qcar as controller  # The brain

# REMOVED: import V2X

# --- Existing Imports ---
from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR
from pal.utilities.math import wrap_to_pi
from hal.content.qcar_functions import QCarEKF
from hal.products.mats import SDCSRoadMap
from qvl.multi_agent import readRobots
from qvl.real_time import QLabsRealTime

# endregion

# REMOVED: TRAFFIC_LIGHTS_CONFIG

# region : Experiment Configuration (remains the same)
tf = 6000
startDelay = 1
controllerUpdateRate = 100
v_ref = 0.4
K_p = 0.1
K_i = 1
enableSteeringControl = True
K_stanley = 1
nodeSequence = [10, 2, 4, 14, 20, 22, 10]
# endregion
traffic_lights = [
    {
        "id": 1,
        "location": [23.667, 9.893, 0.005],
        "rotation": [0, 0, 0],
        "traffic_light_obj": None,
    },
    {
        "id": 3,
        "location": [-21.122, 9.341, 0.005],
        "rotation": [0, 0, 180],
        "traffic_light_obj": None,
    },
]
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


# region : Initial Setup (remains mostly the same)
if enableSteeringControl:
    roadmap = SDCSRoadMap(leftHandTraffic=False)
    waypointSequence = roadmap.generate_path(nodeSequence)
    initialPose = roadmap.get_node_pose(nodeSequence[0]).squeeze()
else:
    initialPose = [0, 0, 0]

if not IS_PHYSICAL_QCAR:
    robotsDir = readRobots()
    Car1 = robotsDir["QC2_0"]
    calibrate = False
else:
    calibrate = "y" in input("do you want to recalibrate?(y/n)")

calibrationPose = [0, 2, -np.pi / 2]

global KILL_THREAD
KILL_THREAD = False


def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True
    perception_module.KILL_THREAD = True


signal.signal(signal.SIGINT, sig_handler)
# endregion


# region : Controller Classes (SpeedController, SteeringController - no changes)
class SpeedController:

    def __init__(self, kp=0, ki=0):
        self.maxThrottle = 0.3

        self.kp = kp
        self.ki = ki

        self.ei = 0

    # ==============  SECTION A -  Speed Control  ====================
    def update(self, v, v_ref, dt):

        e = v_ref - v
        self.ei += dt * e

        return np.clip(
            self.kp * e + self.ki * self.ei, -self.maxThrottle, self.maxThrottle
        )

        return 0


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

    # ==============  SECTION B -  Steering Control  ====================
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

        return 0


def controlLoop(command_queue, shared_pose):
    # region controlLoop setup
    global KILL_THREAD
    u = 0
    delta = 0
    # used to limit data sampling to 10hz
    countMax = controllerUpdateRate / 10
    count = 0
    # endregion

    # region Controller initialization
    speedController = SpeedController(kp=K_p, ki=K_i)
    if enableSteeringControl:
        steeringController = SteeringController(waypoints=waypointSequence, k=K_stanley)
    # endregion

    # region QCar interface setup
    qcar = QCar(readMode=1, frequency=controllerUpdateRate, hilPort=Car1["hilPort"])
    if enableSteeringControl or calibrate:
        ekf = QCarEKF(x_0=initialPose)
        gps = QCarGPS(
            initialPose=calibrationPose,
            calibrate=calibrate,
            gpsPort=Car1["gpsPort"],
            lidarIdealPort=Car1["lidarIdealPort"],
        )
    else:
        gps = memoryview(b"")
    # endregion
    effective_v_ref = v_ref
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
                                command_queue.put("STOP")
                                has_stopped_at[name] = True
                                print(f"Stopping at {name} due to RED light!")
                            elif (
                                traffic_light_status == "GREEN" and has_stopped_at[name]
                            ):
                                command_queue.put("GO")  # GO (resume cruise speed)
                                has_stopped_at[name] = False
                                print(
                                    f"Traffic light at {name} turned GREEN. Resuming movement."
                                )
                        if not inside and has_stopped_at[name]:
                            # Reset stop flag once we leave the area
                            has_stopped_at[name] = False
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
                shared_pose["x"] = ekf.x_hat[0, 0]
                shared_pose["y"] = ekf.x_hat[1, 0]
                shared_pose["th"] = ekf.x_hat[2, 0]  # Share heading (theta)
                shared_pose["v"] = qcar.motorTach  # Share current velocity
                x = ekf.x_hat[0, 0]
                y = ekf.x_hat[1, 0]
                th = ekf.x_hat[2, 0]
                p = np.array([x, y]) + np.array([np.cos(th), np.sin(th)]) * 0.2
            v = qcar.motorTach
            # endregion
            if not command_queue.empty():
                command = command_queue.get()
                if command == "STOP":
                    effective_v_ref = 0.0
                elif command.startswith("GO") and command != "GO":
                    value_str = command[3:]
                    # Convert the extracted string to a float
                    effective_v_ref = float(value_str)  # Restore original speed
                elif command == "GO":
                    effective_v_ref = v_ref
            # region : Update controllers and write to car
            if t < startDelay:
                u = 0
                delta = 0
            else:
                # region : Speed controller update
                u = speedController.update(v, effective_v_ref, dt)
                # endregion

                # region : Steering controller update
                if enableSteeringControl:
                    delta = steeringController.update(p, th, v)
                else:
                    delta = 0
                # endregion

            qcar.write(u, delta)
            # endregion
            continue
        qcar.read_write_std(throttle=0, steering=0)


if __name__ == "__main__":
    # --- Setup multiprocessing queues and shared memory ---
    mp.set_start_method("spawn", force=True)
    perception_queue = mp.Queue(maxsize=1)
    # REMOVED: v2x_queue = mp.Queue(maxsize=1)
    command_queue = mp.Queue(maxsize=1)

    # Shared dictionary for car's pose
    manager = mp.Manager()
    shared_pose = manager.dict({"x": 0.0, "y": 0.0, "th": 0.0, "v": 0.0})

    # --- Start All Processes and Threads ---

    # 1. Perception Module (Eyes) - runs in a separate thread
    perception_proc = Thread(target=run_perception, args=(perception_queue, 0))
    perception_proc.start()

    # REMOVED: V2X Module process

    # 2. Controller Module (Brain) - runs in a separate process
    # MODIFIED: Arguments no longer include v2x_queue
    controller_proc = mp.Process(
        target=controller.main, args=(perception_queue, command_queue)
    )
    controller_proc.start()

    # 3. Main Control Loop (Hands) - runs in a separate thread
    control_thread = Thread(target=controlLoop, args=(command_queue, shared_pose))
    control_thread.start()

    try:
        while control_thread.is_alive() and (not KILL_THREAD):
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutdown initiated by user.")
    finally:
        KILL_THREAD = True
        # Cleanly terminate all processes
        # REMOVED: if v2x_proc.is_alive(): v2x_proc.terminate()
        if controller_proc.is_alive():
            controller_proc.terminate()
        perception_proc.join()
        control_thread.join()
        # REMOVED: v2x_proc.join()
        controller_proc.join()

    if not IS_PHYSICAL_QCAR:
        QLabsRealTime().terminate_all_real_time_models()

    print("Experiment complete.")
