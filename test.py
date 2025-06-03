import ultralytics
import cv2
from ultralytics import solutions

ultralytics.checks()

region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]

# Init ObjectCounter
counter = solutions.ObjectCounter(
    show=True,  # Display the output
    region=region_points,  # Pass region points
    model="yolo11n.pt",  # model="yolo11n-obb.pt" for object counting using YOLO11 OBB model.
    classes=[
        0
    ],  # If you want to count specific classes i.e person and car with COCO pretrained model.
    show_in=True,  # Display in counts
    show_out=True,  # Display out counts
    line_width=2,  # Adjust the line width for bounding boxes and text display
)

cap = cv2.VideoCapture(0)

assert cap.isOpened(), "Error reading video file"
w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)

# # Video writer
# video_writer = cv2.VideoWriter("counting.avi",
#                                cv2.VideoWriter_fourcc(*"mp4v"),
#                                fps, (w, h))

# Process video
import time
import numpy as np

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
    # video_writer.write(results.plot_im)   # write the video frames

    counting_data = np.append(
        counting_data, [int(elapsed_time), int(results.total_tracks)]
    )

    print(results.total_tracks)

    elapsed_time += 1

    np.savez_compressed("counting_data", x=counting_data)

    time.sleep(1)


# cap.release()   # Release the capture
# video_writer.release()
cv2.destroyAllWindows()
