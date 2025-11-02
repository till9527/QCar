This does not include the directory model and the file best.pt inside of it. You will need to train your own model for that. 

Make sure Quanser Interactive Labs is open and cityscape is selected, and then run "run.bat" and it will do everything

Note:

The controller code is not perfect. For the moment, it is programmed to stop at stop signs for 5 seconds regardless of circumstance, and stop at yield signs for 3 seconds regardless of circumstance.

The v2x_helpers.py is a standalone script to run just v2x on both QCars. The main program run from "run.bat" utilizes both perception and v2x.

Additionally, for the controller to properly work each of your roboflow classes must be labeled exactly as such (case sensitive):

"green_light"

"pedestrian"

"Qcar"

"red_light"

"stop_sign"

"yield_sign"
