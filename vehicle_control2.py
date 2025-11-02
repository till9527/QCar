# vehicle_control.py (Refactored "Body")

# region : Imports
import os
import signal
import numpy as np
import time
import multiprocessing as mp
from threading import Thread

# --- REMOVED QLabs modules ---

# --- Existing Imports ---
from perception_module import run_perception
import perception_module
import controller_qcar as controller  # The brain

from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR
from pal.utilities.math import wrap_to_pi
from hal.content.qcar_functions import QCarEKF
from hal.products.mats import SDCSRoadMap
from qvl.multi_agent import readRobots
from qvl.real_time import QLabsRealTime

# endregion

# region : Experiment Configuration (remains the same)
tf = 6000
startDelay = 1
controllerUpdateRate = 100
v_ref = 0.4
K_p = 0.1
K_i = 1
enableSteeringControl = True
K_stanley = 1
nodeSequence = [4, 6, 13, 19, 17, 20, 22, 10, 2, 4]
# endregion

# --- REMOVED: V2X Configuration section ---
# --- REMOVED: V2X Helper Functions section ---

# region : Initial Setup
if enableSteeringControl:
    roadmap = SDCSRoadMap(leftHandTraffic=False)
    waypointSequence = roadmap.generate_path(nodeSequence)
    initialPose = roadmap.get_node_pose(nodeSequence[0]).squeeze()
else:
    initialPose = [0, 0, 0]

if not IS_PHYSICAL_QCAR:
    robotsDir = readRobots()
    Car2 = robotsDir["QC2_1"]
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
    # ... (no changes) ...
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
    # ... (no changes) ...
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
    qcar = QCar(readMode=1, frequency=controllerUpdateRate, hilPort=Car2["hilPort"])
    if enableSteeringControl or calibrate:
        ekf = QCarEKF(x_0=initialPose)
        gps = QCarGPS(
            initialPose=calibrationPose,
            calibrate=calibrate,
            gpsPort=Car2["gpsPort"],
            lidarIdealPort=Car2["lidarIdealPort"],
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

                # --- SHARE POSE ---
                shared_pose["x"] = ekf.x_hat[0, 0]
                shared_pose["y"] = ekf.x_hat[1, 0]
                shared_pose["th"] = ekf.x_hat[2, 0]
                shared_pose["v"] = qcar.motorTach
                # --- END SHARE POSE ---

                x = ekf.x_hat[0, 0]
                y = ekf.x_hat[1, 0]
                th = ekf.x_hat[2, 0]
                p = np.array([x, y]) + np.array([np.cos(th), np.sin(th)]) * 0.2
            v = qcar.motorTach
            # endregion

            # --- Check for commands from the controller "brain" ---
            if not command_queue.empty():
                command = command_queue.get()
                if command == "STOP":
                    effective_v_ref = 0.0
                elif command.startswith("GO") and command != "GO":
                    value_str = command[3:]
                    effective_v_ref = float(value_str)
                elif command == "GO":
                    effective_v_ref = v_ref
            # endregion

            # region : Update controllers and write to car
            if t < startDelay:
                u = 0
                delta = 0
            else:
                u = speedController.update(v, effective_v_ref, dt)
                if enableSteeringControl:
                    delta = steeringController.update(p, th, v)
                else:
                    delta = 0
            qcar.write(u, delta)
            # endregion
            continue
        qcar.read_write_std(throttle=0, steering=0)


if __name__ == "__main__":
    try:
        # --- Setup multiprocessing queues and shared memory ---
        mp.set_start_method("spawn", force=True)
        perception_queue = mp.Queue(maxsize=1)
        command_queue = mp.Queue(maxsize=1)

        manager = mp.Manager()
        shared_pose = manager.dict({"x": 0.0, "y": 0.0, "th": 0.0, "v": 0.0})

        # --- REMOVED: QLabs Setup ---
        # --- REMOVED: V2X Shared State Setup ---

        # --- Start All Processes and Threads ---

        # 1. Perception Module (Eyes)
        perception_proc = Thread(target=run_perception, args=(perception_queue, 1))
        perception_proc.start()

        # --- REMOVED: V2X Status Thread ---

        # 3. Controller Module (Brain)
        # MODIFIED: Args are now much simpler
        controller_proc = mp.Process(
            target=controller.main,
            args=(
                perception_queue,
                command_queue,
                shared_pose,
            ),
        )
        controller_proc.start()

        # 4. Main Control Loop (Hands)
        control_thread = Thread(target=controlLoop, args=(command_queue, shared_pose))
        control_thread.start()

        try:
            while control_thread.is_alive() and (not KILL_THREAD):
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Shutdown initiated by user.")
    finally:
        print("Initiating shutdown...")
        KILL_THREAD = True
        
        if controller_proc.is_alive():
            controller_proc.terminate()
        perception_proc.join()
        # --- REMOVED: statusThread.join() ---
        control_thread.join()
        controller_proc.join()
        print("✅ All threads and processes joined.")

        # --- REMOVED: QLabs close logic ---

        if not IS_PHYSICAL_QCAR:
            # This is still useful to clean up the simulation environment
            QLabsRealTime().terminate_all_real_time_models()

    print("Experiment complete.")
