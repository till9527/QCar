from ultralytics import YOLO

# === CONFIGURATION ===
DATA_YAML = r"C:\Users\user\Downloads\Actual QCar Video.v3i.yolov8\data.yaml"  # <-- updated local path
PROJECT_DIR = (
    r"C:\Users\user\Downloads\Actual QCar Video.v3i.yolov8"  # optional save path
)
MODEL_SIZE = "yolov8n.pt"  # <-- upgraded from 'yolov8n.pt' for better accuracy on GPU

# === LOAD BASE MODEL ===
model = YOLO(MODEL_SIZE)

# === START TRAINING ===
model.train(
    data=DATA_YAML,
    epochs=100,
    imgsz=640,
    batch=16,
    name="physical_model",
    project=PROJECT_DIR,
    workers=0,
    device=0,
    verbose=True,
    patience=20,
    optimizer="Adam",
    lr0=0.001,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    translate=0.1,
    scale=0.5,
)
####  ######     ######## ##     ## ####  ######     ######## ##     ## ######## ##    ##    ##       ########  ######      ###    ##
##  ##    ##       ##    ##     ##  ##  ##    ##    ##       ##     ## ##       ###   ##    ##       ##       ##    ##    ## ##   ##
##  ##             ##    ##     ##  ##  ##          ##       ##     ## ##       ####  ##    ##       ##       ##         ##   ##  ##
##   ######        ##    #########  ##   ######     ######   ##     ## ######   ## ## ##    ##       ######   ##   #### ##     ## ##
##        ##       ##    ##     ##  ##        ##    ##        ##   ##  ##       ##  ####    ##       ##       ##    ##  ######### ##
##  ##    ##       ##    ##     ##  ##  ##    ##    ##         ## ##   ##       ##   ###    ##       ##       ##    ##  ##     ## ##
####  ######        ##    ##     ## ####  ######     ########    ###    ######## ##    ##    ######## ########  ######   ##     ## ########
