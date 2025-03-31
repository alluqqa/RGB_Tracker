import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

def average_rgb(image, box):
    x1, y1, x2, y2 = map(int, box)
    roi = image[y1:y2, x1:x2]

    diff1 = np.abs(roi[:, :, 0] - roi[:, :, 1])
    diff2 = np.abs(roi[:, :, 1] - roi[:, :, 2])
    mask = (diff1 > 5) & (diff2 > 5)
    colored_pixels = roi[mask]

    if colored_pixels.size == 0:
        return (0, 0, 0)

    avg_color = np.mean(colored_pixels, axis=0)
    return tuple(map(int, avg_color))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    if len(results.boxes) > 0:
        # Focus on the largest object detected (any class)
        closest_box = max(
            results.boxes,
            key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1])
        )

        x1, y1, x2, y2 = closest_box.xyxy[0].cpu().numpy()
        label = results.names[int(closest_box.cls)]

        avg_rgb = average_rgb(frame, (x1, y1, x2, y2))

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}: RGB{avg_rgb}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Any Object RGB Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
