#!/usr/bin/env python3
"""
Script to remove all images from dataset_foods source from both the CSV and images folder
"""

import os
import csv
import shutil

# Define paths
CSV_PATH = "csvfiles/latest.csv"
IMAGES_DIR = "images"
BACKUP_CSV = "csvfiles/latest_backup.csv"
REMOVED_IMAGES_DIR = "removed_images"  # A directory to move removed images to (instead of deleting)

# Create backup directory if it doesn't exist
os.makedirs(REMOVED_IMAGES_DIR, exist_ok=True)

# First, read the CSV and identify images to remove
images_to_remove = []
filtered_rows = []

print("Reading CSV file...")
with open(CSV_PATH, 'r') as csvfile:
    # Read the first line to check if it's a comment
    first_line = csvfile.readline().strip()
    # Reset file pointer
    csvfile.seek(0)
    
    # Check if the first line is a comment
    has_comment = first_line.startswith('//')
    
    lines = csvfile.readlines()
    start_idx = 1 if has_comment else 0  # Skip comment line if it exists
    
    # Process the actual CSV content
    for i, line in enumerate(lines):
        if i < start_idx + 1:  # Skip comment and header
            if i == start_idx:
                header = line.strip().split(';')
            continue
            
        row = line.strip().split(';')
        if len(row) >= 2 and row[1] == "dataset_foods":
            images_to_remove.append(row[0])  # Store image name
        else:
            filtered_rows.append(row)  # Keep rows that aren't from dataset_foods
    
print(f"Found {len(images_to_remove)} images from dataset_foods to remove.")

# Create a backup of the original CSV
shutil.copy2(CSV_PATH, BACKUP_CSV)
print(f"Created backup of CSV at {BACKUP_CSV}")

# Write the filtered CSV back
print("Writing filtered CSV...")
with open(CSV_PATH, 'w', newline='') as csvfile:
    # Write the comment line if it exists
    if has_comment:
        csvfile.write(first_line + '\n')
    
    # Write the header
    csvfile.write(';'.join(header) + '\n')
    
    # Write the filtered data
    for row in filtered_rows:
        csvfile.write(';'.join(row) + '\n')
    print(f"Removed {len(images_to_remove)} entries from the CSV file.")

# Now remove the corresponding images (move them to backup folder instead of deleting)
removed_count = 0
not_found_count = 0
for image in images_to_remove:
    image_path = os.path.join(IMAGES_DIR, image)
    if os.path.exists(image_path):
        backup_path = os.path.join(REMOVED_IMAGES_DIR, image)
        shutil.move(image_path, backup_path)
        removed_count += 1
    else:
        not_found_count += 1
        print(f"Warning: Image not found: {image_path}")

print(f"Removed {removed_count} images from {IMAGES_DIR} directory")
if not_found_count > 0:
    print(f"Note: {not_found_count} images listed in CSV were not found in the images directory.")

print("Done!")
