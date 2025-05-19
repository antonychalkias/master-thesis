#!/usr/bin/env python3
"""
Script to test specific images with the trained model.
"""

import os
import sys
import torch
import argparse
from test_model import load_model, process_single_image

# List of random images to test
RANDOM_IMAGES = [
    "20231003_163414.jpg",
    "20230927_102430.jpg",
    "20230927_103046.jpg",
    "20231003_160638.jpg",
    "20231003_162304.jpg",
    "20230927_102750.jpg",
    "20231003_155550.jpg",
    "20230927_102936.jpg",
    "20231003_155349.jpg",
    "20230927_102425.jpg"
]

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test model on specific images")
    parser.add_argument("--model_path", type=str, 
                        default="/Users/chalkiasantonios/Desktop/master-thesis/TRAIN_RESULTS/best_model.pth",
                        help="Path to trained model")
    parser.add_argument("--images_dir", type=str,
                        default="/Users/chalkiasantonios/Desktop/master-thesis/image_set_2",
                        help="Directory containing images")
    parser.add_argument("--output_dir", type=str,
                        default="/Users/chalkiasantonios/Desktop/master-thesis/results_10_random",
                        help="Directory to save results")
    parser.add_argument("--num_classes", type=int, default=11, 
                        help="Number of classes in the model")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU usage")
    
    args = parser.parse_args()
    
    # Set device based on availability
    if args.cpu:
        device = 'cpu'
    else:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'  # Use MPS on macOS
        else:
            device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, idx_to_label = load_model(args.model_path, args.num_classes, device)
    
    # Process each specific image
    for i, img_filename in enumerate(RANDOM_IMAGES):
        img_path = os.path.join(args.images_dir, img_filename)
        
        if os.path.exists(img_path):
            print(f"\nProcessing image {i+1}/10: {img_filename}")
            process_single_image(model, img_path, idx_to_label, device, args.output_dir)
        else:
            print(f"Image not found: {img_path}")
    
    print(f"\nAll results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
