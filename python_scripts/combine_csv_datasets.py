#!/usr/bin/env python3
"""
Script to combine data from labels_for_ordered_dataset.csv and labels_for_dataset_foods.csv 
into a new combined CSV file.
"""

import os
import csv
import json
import pandas as pd
from urllib.parse import unquote, urlparse

# Define paths
base_dir = '/Users/chalkiasantonios/Desktop/master-thesis'
ordered_dataset_csv = os.path.join(base_dir, 'csvfiles', 'labels_for_ordered_dataset.csv')
foods_dataset_csv = os.path.join(base_dir, 'csvfiles', 'labels_for_dataset_foods.csv')
output_csv = os.path.join(base_dir, 'csvfiles', 'combined_dataset_labels.csv')

def extract_image_name_from_url(url):
    """Extract the image filename from a URL string."""
    if not url:
        return None
    
    parsed_url = urlparse(url)
    path = unquote(parsed_url.path)
    filename = os.path.basename(path)
    return filename

def extract_labels_from_json(labels_json):
    """Extract food labels from the JSON annotation data."""
    try:
        if not labels_json:
            return []
        
        labels_data = json.loads(labels_json)
        food_labels = []
        
        for item in labels_data:
            if "polygonlabels" in item and item["polygonlabels"]:
                food_labels.extend(item["polygonlabels"])
        
        # Remove duplicates while preserving order
        unique_labels = []
        for label in food_labels:
            if label not in unique_labels:
                unique_labels.append(label)
        
        return unique_labels
    except json.JSONDecodeError:
        print(f"Error decoding JSON: {labels_json[:50]}...")
        return []

def combine_csv_files():
    """Combine the two CSV files into one comprehensive dataset."""
    print("Loading datasets...")
    
    # Load the ordered dataset
    ordered_df = pd.read_csv(ordered_dataset_csv)
    print(f"Loaded {len(ordered_df)} records from ordered dataset")
    
    # Load the foods dataset
    foods_df = pd.read_csv(foods_dataset_csv)
    print(f"Loaded {len(foods_df)} records from foods dataset")
    
    # Extract image names and labels from foods_df
    foods_df['image_name'] = foods_df['image'].apply(extract_image_name_from_url)
    foods_df['food_labels'] = foods_df['label'].apply(extract_labels_from_json)
    foods_df['total_weight'] = foods_df['totalWeight']
    
    # Create a new DataFrame for the combined data
    combined_data = []
    
    # Process ordered dataset entries
    print("Processing ordered dataset entries...")
    for _, row in ordered_df.iterrows():
        entry = {
            'image_name': row['image'],
            'source': 'ordered_dataset',
            'labels': row['label'],
            'weight': row['weight'],
            'volume': row.get('volume', None),
            'energy': row.get('energy', None),
            'total_weight': row['weight'],  # For ordered dataset, item weight is total weight
            'annotation_id': None,
            'annotator': None,
            'lead_time': None
        }
        combined_data.append(entry)
    
    # Process foods dataset entries
    print("Processing foods dataset entries...")
    for _, row in foods_df.iterrows():
        entry = {
            'image_name': row['image_name'],
            'source': 'dataset_foods',
            'labels': ','.join(row['food_labels']) if row['food_labels'] else None,
            'weight': None,  # Individual food weights not available
            'volume': None,
            'energy': None,
            'total_weight': row['total_weight'],
            'annotation_id': row['annotation_id'],
            'annotator': row['annotator'],
            'lead_time': row['lead_time']
        }
        combined_data.append(entry)
    
    # Convert to DataFrame
    combined_df = pd.DataFrame(combined_data)
    
    # Save to CSV
    combined_df.to_csv(output_csv, index=False)
    print(f"Successfully combined datasets into {output_csv}")
    print(f"Total records in combined dataset: {len(combined_df)}")

if __name__ == "__main__":
    print("Starting to combine dataset CSV files...")
    combine_csv_files()
    print("Process complete!")
