#!/usr/bin/env python3
"""
Data processing and loading for food recognition and weight estimation model.
"""

from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
import torch
import os
import pandas as pd

class FoodDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        # Cache for found paths to speed up loading
        self.path_cache = {}
    
    def __len__(self):
        return len(self.df)
    
    def _try_load_image(self, path, row):
        """Helper to attempt loading an image from a path"""
        try:
            image = Image.open(path).convert('RGB')
            image = self.transform(image)
            return image, torch.tensor(row['label_idx'], dtype=torch.long), torch.tensor(row['weight'], dtype=torch.float32)
        except Exception:
            return None
    
    def _create_placeholder(self, row):
        """Create a placeholder black image for missing files"""
        image = Image.new('RGB', (224, 224), color='black')
        image = self.transform(image)
        return image, torch.tensor(row['label_idx'], dtype=torch.long), torch.tensor(row['weight'], dtype=torch.float32)
    
    def _find_image_path(self, img_name):
        """Find the correct path for an image, handling different cases and extensions"""
        # Get base name without extension
        name_without_ext, _ = os.path.splitext(img_name)
        
        # 1. Try original path first
        original_path = os.path.join(self.image_dir, img_name)
        if os.path.exists(original_path):
            return original_path
            
        # 2. Try with common extensions
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            test_path = os.path.join(self.image_dir, name_without_ext + ext)
            if os.path.exists(test_path):
                return test_path
        
        # 3. Case-insensitive search
        try:
            name_lower = name_without_ext.lower()
            for file in os.listdir(self.image_dir):
                file_name, _ = os.path.splitext(file)
                if file_name.lower() == name_lower:
                    return os.path.join(self.image_dir, file)
        except Exception as e:
            print(f"Error during file search: {e}")
            
        # Not found
        return None
    
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            img_name = row['image_name']
            
            # Check cache first
            if img_name in self.path_cache:
                cached_path = self.path_cache[img_name]
                if cached_path == "PLACEHOLDER":
                    return self._create_placeholder(row)
                    
                result = self._try_load_image(cached_path, row)
                if result is not None:
                    return result
                # Path no longer valid, clear from cache
                del self.path_cache[img_name]
            
            # Try to find the image path
            img_path = self._find_image_path(img_name)
            
            if img_path:
                # Found a path, try to load it
                result = self._try_load_image(img_path, row)
                if result is not None:
                    self.path_cache[img_name] = img_path
                    return result
            
            # If we get here, image wasn't found or couldn't be loaded
            print(f"Warning: Image {img_name} not found or corrupted, using placeholder")
            self.path_cache[img_name] = "PLACEHOLDER"
            return self._create_placeholder(row)
            
        except Exception as e:
            print(f"Unexpected error for index {idx}: {e}")
            return self._create_placeholder(row)

def prepare_data(csv_path, images_dir, batch_size=16, num_workers=0):
    """
    Prepare data for training and validation
    
    Args:
        csv_path: Path to CSV file with annotations
        images_dir: Path to directory with images
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for data loading
    
    Returns:
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        label_to_idx: Dictionary mapping labels to indices
    """
    # Load and process CSV
    try:
        df = pd.read_csv(csv_path, sep=';', quotechar='"')
        print(f"Successfully loaded {len(df)} records from {csv_path}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise
        
    # Create label-to-index mapping
    label_to_idx = {label: idx for idx, label in enumerate(df['labels'].unique())}
    df['label_idx'] = df['labels'].map(label_to_idx)

    print(f"Number of classes: {len(label_to_idx)}")
    print(df.head())

    # Create dataset
    dataset = FoodDataset(df, images_dir)
    print(f"Dataset size: {len(dataset)} images")

    # Split data into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=False
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=False
    )

    print(f"Training on {train_size} samples, validating on {val_size} samples")

    return train_dataloader, val_dataloader, label_to_idx
