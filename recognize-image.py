import cv2
import numpy as np
from datetime import datetime
import csv
import os
import glob

# Define folders
input_folder = './imagens'
output_folder = './resultados'
log_path = os.path.join(output_folder, 'germination_log.csv')

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get all image files (jpg, jpeg, png) from input folder
image_paths = glob.glob(os.path.join(input_folder, '*.[jJpP][pPnN][gG]'))

# Open CSV log file
with open(log_path, 'a') as f:
    writer = csv.writer(f)
    
    for image_path in image_paths:
        # Get filename (e.g., solo2.jpg)
        filename = os.path.basename(image_path)
        date_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not read {filename}, skipping.")
            continue

        # Resize
        image = cv2.resize(image, (640, 480))

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Green mask
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Morphological filtering
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Contour detection
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Count valid contours
        germinated_count = sum(cv2.contourArea(c) > 100 for c in contours)

        # Log result
        writer.writerow([date_stamp, filename, germinated_count])
        print(f"{filename}: {germinated_count} germinated seeds detected.")
