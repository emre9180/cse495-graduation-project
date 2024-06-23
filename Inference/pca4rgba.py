import os
import numpy as np
import cv2
from sklearn.decomposition import PCA

def pca_transform_image(image_path):
    # Load the RGBA image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Ensure alpha channel is included

    # Check if image is loaded properly
    if image is None:
        raise ValueError("Image not found or path is incorrect")

    # Reshape the image to 2D (pixels by channels)
    pixels, width, height = image.shape[0], image.shape[1], image.shape[2]
    image_reshaped = image.reshape(pixels * width, height)

    # Apply PCA to reduce to 3 channels
    pca = PCA(n_components=3)
    image_transformed = pca.fit_transform(image_reshaped)

    # Reshape back to 3-channel image
    image_3_channel = image_transformed.reshape(pixels, width, 3)

    # Convert back to uint8
    return np.clip(image_3_channel, 0, 255).astype('uint8')

def process_images(input_directory, output_directory):
    # Check if output directory exists, if not, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Process each image in the input directory
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, filename)

            try:
                transformed_image = pca_transform_image(input_path)
                cv2.imwrite(output_path, transformed_image)
                print(f"Processed and saved: {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Example usage
input_directory = r'C:\Users\alicakici\Desktop\rgba\test\images'  # Replace with your input directory path
output_directory = r'C:\Users\alicakici\Desktop\pca-rgba\test\images'  # Replace with your output directory path

process_images(input_directory, output_directory)
