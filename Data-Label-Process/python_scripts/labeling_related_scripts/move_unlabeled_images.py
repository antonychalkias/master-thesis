#!/usr/bin/env python3
"""
Script to identify unlabeled images in dataset_foods and move them to a new directory.
"""
import os
import csv
import shutil
import re
from urllib.parse import unquote, urlparse

# Define paths
base_dir = '/Users/chalkiasantonios/Desktop/master-thesis'
csv_file_path = os.path.join(base_dir, 'csvfiles', 'labels_for_dataset_foods.csv')
images_dir = os.path.join(base_dir, 'dataset_foods')
unlabeled_dir = os.path.join(base_dir, 'dataset_foods_unlabeled')

# Create directory for unlabeled images if it doesn't exist
if not os.path.exists(unlabeled_dir):
    os.makedirs(unlabeled_dir)
    print(f"Created directory: {unlabeled_dir}")

# Function to extract filenames from URLs in CSV
def get_labeled_image_names():
    labeled_images = set()
    
    with open(csv_file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_url = row.get('image')
            if image_url:
                # Extract filename from URL
                parsed_url = urlparse(image_url)
                path = unquote(parsed_url.path)
                filename = os.path.basename(path)
                labeled_images.add(filename)
    
    print(f"Found {len(labeled_images)} labeled images in CSV file")
    return labeled_images

# Get all image files in the dataset_foods directory
def get_all_images():
    all_images = []
    for filename in os.listdir(images_dir):
        # Only consider image files
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
            all_images.append(filename)
    
    print(f"Found {len(all_images)} total image files in {images_dir}")
    return all_images

# Move unlabeled images to the new directory
def move_unlabeled_images(labeled_images, all_images):
    unlabeled_count = 0
    
    for image in all_images:
        if image not in labeled_images:
            src_path = os.path.join(images_dir, image)
            dst_path = os.path.join(unlabeled_dir, image)
            
            try:
                shutil.move(src_path, dst_path)
                unlabeled_count += 1
                print(f"Moved: {image}")
            except Exception as e:
                print(f"Error moving {image}: {e}")
    
    print(f"\nSummary: Moved {unlabeled_count} unlabeled images to {unlabeled_dir}")

if __name__ == "__main__":
    print("Starting process to identify and move unlabeled images...")
    
    labeled_images = get_labeled_image_names()
    all_images = get_all_images()
    
    move_unlabeled_images(labeled_images, all_images)
    
    print("Process complete!")
