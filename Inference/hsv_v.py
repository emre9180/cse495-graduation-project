import os
import cv2
import numpy as np

# Replace these paths with your actual paths
path_to_images = r'C:\Users\alicakici\Downloads\alpha\RGB-PEPPER-DATASET\all-train-test-valid\images'  # e.g., '/path/to/input/images'
path_to_save_images = r'C:\Users\alicakici\Downloads\alpha\RGB-PEPPER-DATASET\all-train-test-valid\asv'  # e.g., '/path/to/output/images'

def process_and_save_images(input_path, output_path):
    # Check if output directory exists, if not create it
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for filename in os.listdir(input_path):
        if filename.endswith(".png"):  # Assuming the images are in PNG format
            # Read the image with RGBA channels
            image_path = os.path.join(input_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            # Split into channels
            r, g, b, a = cv2.split(image)

            # Convert RGB to HSV
            hsv_image = cv2.cvtColor(cv2.merge([r, g, b]), cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv_image)

            # Replace the V channel with Alpha
            hsv_modified = cv2.merge([a, s, v])

            # Convert back to RGBA for saving
            # rgb_modified = cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2RGB)
            # rgba_modified = cv2.merge([rgb_modified[:, :, 0], rgb_modified[:, :, 1], rgb_modified[:, :, 2], a])

            # Save the image
            cv2.imwrite(os.path.join(output_path, filename), hsv_modified)

# Run the function
process_and_save_images(path_to_images, path_to_save_images)
