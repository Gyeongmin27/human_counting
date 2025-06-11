import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
import cv2

# Load model
device = select_device("")
model = attempt_load("weights/yolov5s.pt", map_location=device)
model.eval()

# Open CSI camera
cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Preprocess and inference
    img = cv2.resize(frame, (640, 640))
    img = img[..., ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img)[0]
    pred = non_max_suppression(pred, 0.25, 0.45, classes=[0])  # class 0 = person

    # Count people
    count = 0
    for det in pred:
        if det is not None and len(det):
            count += len(det)
            for *xyxy, conf, cls in det:
                label = f"person {conf:.2f}"
                cv2.rectangle(
                    frame,
                    (int(xyxy[0]), int(xyxy[1])),
                    (int(xyxy[2]), int(xyxy[3])),
                    (255, 0, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    label,
                    (int(xyxy[0]), int(xyxy[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )
    cv2.putText(
        frame, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )
    cv2.imshow("Human Counting", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
