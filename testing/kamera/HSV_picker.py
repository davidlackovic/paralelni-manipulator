import cv2
import numpy as np

# Load the uploaded image
image_path = 'testing/kamera/zogica_10.png'
image = cv2.imread(image_path)

# Convert the image to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Compute the mean HSV value of the image
mean_hsv = cv2.mean(hsv_image)[:3]

# Suggest HSV range around the mean
hue, saturation, value = mean_hsv
lower_hsv = np.array([max(hue - 20, 0), max(saturation - 50, 0), max(value - 50, 0)], dtype=np.uint8)
upper_hsv = np.array([min(hue + 20, 179), min(saturation + 50, 255), min(value + 50, 255)], dtype=np.uint8)

print(f"Lower: {lower_hsv}, upper: {upper_hsv}")
