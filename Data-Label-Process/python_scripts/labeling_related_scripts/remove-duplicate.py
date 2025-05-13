#!/usr/bin/env python3
import os
import hashlib
from pathlib import Path
import shutil
import argparse

def calculate_hash(file_path):
    """Calculate the MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_image_files(directory):
    """Get all image files in the directory"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(directory).glob(f'**/*{ext}')))
        image_files.extend(list(Path(directory).glob(f'**/*{ext.upper()}')))
    return image_files

def handle_duplicates(duplicates, move_to=None, dry_run=False):
    """Handle the duplicate files based on the specified action"""
    if dry_run:
        print("Dry run mode - no files were moved or deleted")
        if duplicates:
            print("The following duplicates were found:")
            for dup in duplicates:
                print(f"  {dup}")
        return
        
    for duplicate in duplicates:
        if move_to:
            # Move the duplicate to the destination directory
            shutil.move(duplicate, os.path.join(move_to, os.path.basename(duplicate)))
            print(f"Moved {duplicate} to {move_to}")
        else:
            # Remove the duplicate
            os.remove(duplicate)
            print(f"Removed {duplicate}")

def find_and_remove_duplicates(directory, move_to=None, dry_run=False):
    """Find and remove duplicate files based on their hash values"""
    print(f"Scanning directory: {directory}")
    
    # Create the destination directory if it doesn't exist and move_to is specified
    if move_to and not dry_run:
        os.makedirs(move_to, exist_ok=True)
        print(f"Duplicates will be moved to: {move_to}")
    
    # Dictionary to store file hash and path
    files_dict = {}
    # List to store duplicate files
    duplicates = []
    
    # Get all image files
    image_files = get_image_files(directory)
    total_files = len(image_files)
    print(f"Found {total_files} image files")
    
    # Calculate hash for each file and check for duplicates
    for index, file_path in enumerate(image_files, 1):
        if index % 100 == 0:
            print(f"Processing file {index}/{total_files}")
        
        file_hash = calculate_hash(file_path)
        
        if file_hash in files_dict:
            # This is a duplicate
            duplicates.append(str(file_path))
            print(f"Found duplicate: {file_path} is same as {files_dict[file_hash]}")
        else:
            # This is a new file
            files_dict[file_hash] = str(file_path)
    
    print(f"\nFound {len(duplicates)} duplicate files out of {total_files} total files")
    
    # Handle the duplicates
    handle_duplicates(duplicates, move_to, dry_run)
    
    return duplicates

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find and remove duplicate image files based on their hash values")
    parser.add_argument("directory", help="Directory to scan for duplicates")
    parser.add_argument("--move-to", help="Directory to move duplicates to instead of deleting them", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Only report duplicates without moving or deleting")
    
    args = parser.parse_args()
    
    find_and_remove_duplicates(args.directory, args.move_to, args.dry_run)