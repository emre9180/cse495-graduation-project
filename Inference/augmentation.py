import albumentations as A
import cv2
import os
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
from shutil import copy2

# Define the augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomCrop(width=640, height=640, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.15),
    # Add more transformations as needed
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def augment_image(image_path, label_path, save_dir):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Read YOLO formatted labels
    with open(label_path, 'r') as file:
        lines = file.readlines()
        bboxes = [[float(x) for x in line.split()] for line in lines]
        class_labels = [int(bbox[0]) for bbox in bboxes]
        bboxes = [bbox[1:] for bbox in bboxes]  # Remove class label from bbox

    for i in range(5):
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        aug_image = augmented['image']
        aug_bboxes = augmented['bboxes']

        aug_image_path = os.path.join(save_dir, 'images', f"{os.path.splitext(os.path.basename(image_path))[0]}_aug{i}.png")
        cv2.imwrite(aug_image_path, aug_image)

        # Save augmented labels in YOLO format
        aug_label_path = os.path.join(save_dir, 'labels', f"{os.path.splitext(os.path.basename(label_path))[0]}_aug{i}.txt")
        with open(aug_label_path, 'w') as file:
            for bbox, label in zip(aug_bboxes, class_labels):
                file.write(f"{label} {' '.join(map(str, bbox))}\n")

def split_dataset(image_dir, label_dir, root_dir):
    # Create necessary directories
    os.makedirs(os.path.join(root_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'val', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'test', 'labels'), exist_ok=True)

    # List all images
    all_images = os.listdir(image_dir)

    # Split into train, validation, and test sets (70%, 20%, 10%)
    total_images = len(all_images)
    train_images = all_images[:int(0.7 * total_images)]
    val_images = all_images[int(0.7 * total_images):int(0.9 * total_images)]
    test_images = all_images[int(0.9 * total_images):]

    # Process train images
    for image_name in train_images:
        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + '.txt')
        augment_image(image_path, label_path, os.path.join(root_dir, 'train'))

    # Process validation and test images
    for dataset, images in zip(['val', 'test'], [val_images, test_images]):
        for image_name in images:
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + '.txt')
            copy2(image_path, os.path.join(root_dir, dataset, 'images'))
            copy2(label_path, os.path.join(root_dir, dataset, 'labels'))

# Example usage
rgb = r"C:\Users\alicakici\Downloads\12-ocak-augmentesiz-dataset\11-ocak-augmentesiz-dataset\CSE495 Pepper Detection EGB-3"
red = r"C:\Users\alicakici\Downloads\12-ocak-augmentesiz-dataset\11-ocak-augmentesiz-dataset\CSE495 Red Relabelling V2-3"
rgba = r"C:\Users\alicakici\Downloads\alpha\RGB-PEPPER-DATASET\all-train-test-valid"
hsva = r"C:\Users\alicakici\Downloads\alpha\RGB-PEPPER-DATASET\all-train-test-valid\hsv_a_images"
image_dir = hsva + '\images'
label_dir = hsva + '\labels'
root_dir = r'C:\Users\alicakici\Desktop\hsv-v'
split_dataset(image_dir, label_dir, root_dir)
