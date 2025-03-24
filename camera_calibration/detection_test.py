import cv2
import numpy as np
import glob
# Load the image
images = glob.glob("calibration_photos/calibration_0.jpg")
image = cv2.imread(images[0])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the checkerboard size (7x7 grid = 6x6 inner corners)
CHECKERBOARD = (10, 7)

# Find the checkerboard corners
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

# If found, draw the corners
if ret:
    image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners, ret)
    print("✅ Checkerboard detected!")
else:
    print("❌ Checkerboard NOT detected.")

# Show the image
cv2.imshow("Checkerboard Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
