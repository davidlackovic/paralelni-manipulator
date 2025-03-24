import cv2
import os

# Create the folder if it doesn't exist
SAVE_FOLDER = "calibration_photos"
os.makedirs(SAVE_FOLDER, exist_ok=True)

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Open the camera
cap.set(cv2.CAP_PROP_EXPOSURE, -8)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
photo_count = 0  # Counter for saved images

print("Press SPACE to capture a photo. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    processed_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
    if not ret:
        print("Failed to capture image")
        break

    cv2.imshow("Camera Feed", processed_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # SPACE to capture
        photo_path = os.path.join(SAVE_FOLDER, f"calibration_{photo_count}.jpg")
        cv2.imwrite(photo_path, processed_frame)
        print(f"Saved: {photo_path}")
        photo_count += 1

    elif key == ord('q'):  # 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
