import sys
import time
import math
from threading import Thread

# Quanser Imports
from qvl.qlabs import QuanserInteractiveLabs
from qvl.real_time import QLabsRealTime
from qvl.free_camera import QLabsFreeCamera
from qvl.crosswalk import QLabsCrosswalk
from qvl.traffic_light import QLabsTrafficLight
from qvl.person import QLabsPerson
from qvl.stop_sign import QLabsStopSign
from qvl.yield_sign import QLabsYieldSign

# --- Configuration Constants ---
TRAFFIC_LIGHTS_CONFIG = [
    {"id": 1, "location": [23.667, 9.893, 0.005], "rotation": [0, 0, 0]},
    {"id": 3, "location": [-21.122, 9.341, 0.005], "rotation": [0, 0, 180]},
]
STOP_SIGNS_CONFIG = [
    {"id": 100, "location": [-6.238, 6.47, 0.2], "rotation": [0, 0, 180]},
    {"id": 101, "location": [-2.067, 16.986, 0.215], "rotation": [0, 0, 90]},
    {"id": 102, "location": [7.989, 13.371, 0.215], "rotation": [0, 0, 360]},
    {"id": 103, "location": [4.733, 2.166, 0.215], "rotation": [0, 0, 270]},
]
YIELD_SIGNS_CONFIG = [
    {"id": 200, "location": [25.007, 32.494, 0.2], "rotation": [0, 0, -90]},
    {"id": 201, "location": [5.25, 39.477, 0.215], "rotation": [0, 0, 180]},
    {"id": 202, "location": [11.136, 28.326, 0.215], "rotation": [0, 0, 225]},
]
CROSSWALK_START = [-10.482, 40, 1]
CROSSWALK_END = [-10.774, 47.083, 1]
PEDESTRIAN_ROTATION = [0, 0, math.pi/2]
LOCATION_START_P1 = [-6.8, 40.7, 0.005]





ROTATION_P1P2 = [0,0,math.pi/2]

SCALE = [1,1,1]

ROTATION_P3 = [0,0,90]



LOCATION_END_P1 = [-7.6, 51, 0.005]



CROSSWALK_LOCATION = [-10.788, 45, 0.00]

# The length of the path across the road

CROSSWALK_PATH_LENGTH = 8



# Calculate the start and end points for the pedestrian patrol path

# The crosswalk is rotated 90 degrees, so the path is along the Y-axis

CROSSWALK_START = [-10.482, 40, 1]

CROSSWALK_END = [-10.774, 47.083, 1]



# Pedestrian orientation to face along the path (90 degrees)

PEDESTRIAN_ROTATION = [0, 0, math.pi/2]

PEDESTRIAN_SCALE = [1, 1, 1]



# --- Logic Functions for Threading ---
def traffic_light_sequence(traffic_light, red_time=13, green_time=6, yellow_time=1, delay=0):
    """Controls the R-Y-G sequence for a single traffic light in a continuous loop."""
    time.sleep(delay)
    while True:
        traffic_light.set_color(QLabsTrafficLight.COLOR_RED)
        time.sleep(red_time)
        traffic_light.set_color(QLabsTrafficLight.COLOR_GREEN)
        time.sleep(green_time)
        traffic_light.set_color(QLabsTrafficLight.COLOR_YELLOW)
        time.sleep(yellow_time)

def pedestrian_patrol(person, start_location, end_location, speed):
    """Controls a person to walk back and forth between two points."""
    while True:
        person.move_to(location=end_location, speed=speed, waitForConfirmation=True)
        time.sleep(10)
        person.move_to(location=start_location, speed=speed, waitForConfirmation=True)
        time.sleep(10)

# --- Main Script Execution ---
print("Connecting to QLabs...")
qlabs = QuanserInteractiveLabs()
if not qlabs.open("localhost"):
    print("FATAL: Unable to connect to QLabs. Is the simulation running?")
    sys.exit()
print("Connection successful.")

# == 1. SETUP PHASE: Spawn all actors in the simulation ==
print("Spawning all actors...")

# Spawn Pedestrian
person1 = QLabsPerson(qlabs)
person1.spawn_id(actorNumber=0, location=CROSSWALK_START, rotation=PEDESTRIAN_ROTATION, scale=[1,1,1], configuration=6, waitForConfirmation=True)
crosswalk = QLabsCrosswalk(qlabs)
crosswalk.spawn_id(0, CROSSWALK_LOCATION, [0, 0, math.pi/2], [1, 1, 1], 0, 1)
# Spawn Traffic Lights
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
print(f"Spawned {len(traffic_light_handles)} traffic lights.")

# Spawn Stop Signs
for config in STOP_SIGNS_CONFIG:
    sign = QLabsStopSign(qlabs)
    sign.spawn_id_degrees(
        actorNumber=config["id"], location=config["location"], rotation=config["rotation"],
        scale=[1, 1, 1], configuration=0, waitForConfirmation=True
    )

# Spawn Yield Signs (Corrected indentation)
for config in YIELD_SIGNS_CONFIG:
    sign = QLabsYieldSign(qlabs)
    sign.spawn_id_degrees(
        actorNumber=config["id"], location=config["location"], rotation=config["rotation"],
        scale=[1, 1, 1], configuration=0, waitForConfirmation=True
    )

print("All actors have been spawned.")

# == 2. LOGIC PHASE: Start background threads for continuous actions ==
print("Starting background logic for pedestrian and traffic lights...")

# Start pedestrian logic
Thread(target=pedestrian_patrol, args=(person1, CROSSWALK_START, CROSSWALK_END, person1.WALK)).start()

# Start traffic light logic
for i, light_handle in enumerate(traffic_light_handles):
    Thread(target=traffic_light_sequence, args=(light_handle, 13, 6, 1, i * 5)).start()

print("Environment logic is now running.")

# == 3. KEEP ALIVE PHASE: Prevent the main script from exiting ==
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nScript terminated by user. Closing QLabs connection.")
    qlabs.close()
    sys.exit()