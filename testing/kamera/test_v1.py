import cv2

def open_webcam(camera_index=1):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera at index {camera_index}")
        return

    while True:
        ret, frame = cap.read() 
        if not ret:
            print("Error: Could not read frame")
            break

        cv2.imshow('Webcam Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

open_webcam(1)
