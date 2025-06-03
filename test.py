import ultralytics
import cv2
from ultralytics import solutions
import time
import numpy as np

ultralytics.checks()

region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]

# Init ObjectCounter
counter = solutions.ObjectCounter(
    show=True,
    region=region_points,
    model="yolo11n.pt",
    classes=[0],
    show_in=True,
    show_out=True,
    line_width=2,
)

# --- Jetson Nano Camera Pipeline ---
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )

# Use GStreamer pipeline for CSI camera
cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

assert (
    cap.isOpened()
), "Error: Unable to open Jetson Nano camera. Check camera connection and pipeline."

w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)

# Process video
s = time.time()
counting_data = np.array([s], dtype=int)
elapsed_time = 0

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print(
            "Video frame is empty or video processing has been successfully completed."
        )
        break
    results = counter(im0)  # count the objects

    counting_data = np.append(
        counting_data, [int(elapsed_time), int(results.total_tracks)]
    )

    print(results.total_tracks)

    elapsed_time += 1

    np.savez_compressed("counting_data", x=counting_data)

    time.sleep(1)

cap.release()
cv2.destroyAllWindows()
