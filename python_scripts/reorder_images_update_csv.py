#!/usr/bin/env python3
"""
Script to combine images from dataset_foods and ordered_dataset folders,
rename them sequentially, and update the CSV file with the new image names.
"""

import os
import shutil
import pandas as pd
import glob
from pathlib import Path
import sys

def main():
    # Define paths
    master_thesis_dir = Path('/Users/chalkiasantonios/Desktop/master-thesis')
    dataset_foods_dir = master_thesis_dir / 'dataset_foods'
    ordered_dataset_dir = master_thesis_dir / 'ordered_dataset'
    output_dir = master_thesis_dir / 'ordered_dataset_foods_ready'
    
    # Define CSV paths
    input_csv = master_thesis_dir / 'csvfiles' / 'combined_dataset_labels.csv'
    output_csv = master_thesis_dir / 'csvfiles' / 'combined_dataset_labels_ready.csv'
    
    # Create output directory if it doesn't exist
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Output directory already exists: {output_dir}")
        # Clear any existing files
        for file in output_dir.glob("*"):
            file.unlink()
        print("Cleared existing files from output directory")
    
    # Read the CSV file
    try:
        df = pd.read_csv(input_csv)
        print(f"Successfully loaded CSV with {len(df)} records")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)
    
    # Create a mapping from old image names to new ones
    image_mapping = {}
    file_to_image_mapping = {}
    new_image_counter = 1
    
    # Get lists of images from both directories
    foods_images = list(dataset_foods_dir.glob('*.jpeg')) + list(dataset_foods_dir.glob('*.jpg')) + list(dataset_foods_dir.glob('*.JPG'))
    ordered_images = list(ordered_dataset_dir.glob('*.jpeg')) + list(ordered_dataset_dir.glob('*.jpg')) + list(ordered_dataset_dir.glob('*.JPG'))
    
    all_images = foods_images + ordered_images
    print(f"Found {len(foods_images)} images in dataset_foods directory")
    print(f"Found {len(ordered_images)} images in ordered_dataset directory")
    print(f"Total images to process: {len(all_images)}")
    
    # First, create a reference of existing images in the CSV
    csv_images = set(df['image_name'])
    print(f"Found {len(csv_images)} unique image references in CSV")
    
    # Process each image
    for img_path in all_images:
        # Get original filename
        original_name = img_path.name
        
        # Determine file extension from original file
        _, file_ext = os.path.splitext(original_name)
        
        # Create new filename (keeping the original extension)
        new_name = f"{new_image_counter}{file_ext}"
        
        # Store direct file mapping
        file_to_image_mapping[original_name] = new_name
        
        # Also map any entries in the CSV that might match this file
        # Store mapping for both the full name and without extension
        image_mapping[original_name] = new_name
        name_without_ext = os.path.splitext(original_name)[0]
        image_mapping[name_without_ext] = new_name
        
        # Copy and rename the file
        dest_path = output_dir / new_name
        shutil.copy2(img_path, dest_path)
        print(f"Copied {original_name} to {new_name}")
        
        new_image_counter += 1
    
    # Update the CSV file
    df['original_image_name'] = df['image_name']  # Preserve the original image names
    
    # Create new image names based on the mapping
    def map_image_name(row):
        original_name = row['image_name']
        # Try direct match
        if original_name in image_mapping:
            return image_mapping[original_name]
        
        # Try match with source directory consideration
        source = row['source'] if 'source' in row else ''
        if source == 'dataset_foods':
            # Check if this image exists in dataset_foods
            for img_name in file_to_image_mapping:
                if (dataset_foods_dir / img_name).exists() and (original_name in img_name or img_name in original_name):
                    return file_to_image_mapping[img_name]
        elif source == 'ordered_dataset':
            # Check if this image exists in ordered_dataset
            for img_name in file_to_image_mapping:
                if (ordered_dataset_dir / img_name).exists() and (original_name in img_name or img_name in original_name):
                    return file_to_image_mapping[img_name]
        
        # If not found, try substring match
        for old_name in image_mapping:
            if original_name in old_name or old_name in original_name:
                return image_mapping[old_name]
                
        print(f"Warning: Could not find mapping for image {original_name}")
        return original_name  # If not found, keep original
    
    df['image_name'] = df.apply(map_image_name, axis=1)
    
    # Save the updated CSV
    df.to_csv(output_csv, index=False)
    print(f"Updated CSV saved to {output_csv} with {len(df)} records")
    print(f"Total images processed and renamed: {new_image_counter - 1}")

if __name__ == "__main__":
    main()
