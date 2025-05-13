#!/usr/bin/env python3
"""
Utility functions for food recognition and weight estimation model.
"""

import os
import argparse

def parse_args():
    """
    Parse command-line arguments for the training script.
    
    Returns:
        args: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train food recognition and weight estimation model")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to the CSV file with annotations")
    parser.add_argument("--images_dir", type=str, default=None, help="Path to the directory with images")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model_dir", type=str, default=None, help="Directory to save the model")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    return parser.parse_args()

def setup_paths(args):
    """
    Set up paths for CSV file, images directory, and model save directory.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        args: Updated arguments with default paths set
    """
    # Get project root directory
    master_thesis_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Set default paths if not provided
    if args.csv_path is None:
        args.csv_path = os.path.join(master_thesis_dir, "csvfiles", "latest_lab.csv")
        # Check if file exists, if not try with the small test file
        if not os.path.exists(args.csv_path):
            print(f"Warning: CSV file not found at {args.csv_path}")
            # Try alternative path
            alternative_csv = os.path.join(master_thesis_dir, "csvfiles", "labels_latest_with_4_rows.csv")
            if os.path.exists(alternative_csv):
                print(f"Using alternative CSV file: {alternative_csv}")
                args.csv_path = alternative_csv
    
    if args.images_dir is None:
        args.images_dir = os.path.join(master_thesis_dir, "images")
    
    if args.model_dir is None:
        args.model_dir = os.path.join(master_thesis_dir, "models")
    
    print(f"CSV path: {args.csv_path}")
    print(f"Images directory: {args.images_dir}")
    print(f"Model save directory: {args.model_dir}")
    
    # Create model save directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    return args

def get_device():
    """
    Get the best available device for training (CUDA, MPS, or CPU).
    
    Returns:
        device: PyTorch device
    """
    import torch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device
