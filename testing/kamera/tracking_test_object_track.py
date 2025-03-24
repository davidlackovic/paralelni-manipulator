from ultralytics import YOLO
import cv2

# Load pretrained YOLOv8 model (or train your own for better accuracy)
model = YOLO("yolov8n.pt")  # "n" version is lightweight, use "m" or "l" for better accuracy

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Run YOLO detection
    for r in results:
        for box in r.boxes:  # Iterate through detected objects
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
            print(f"Detected ball at ({x1}, {y1}) to ({x2}, {y2})")

    cv2.imshow("Ball Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
