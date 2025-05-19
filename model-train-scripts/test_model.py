#!/usr/bin/env python3
"""
Modified inference script for testing the best_model.pth model on macOS with GPU support.
"""

import os
import sys
import torch
import argparse
import json
from PIL import Image
import numpy as np
from torchvision import transforms
from model import MultiTaskNet
import matplotlib.pyplot as plt

# Check for MPS (Metal Performance Shaders) support on macOS
if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
    # Initialize MPS
    mps_available = torch.backends.mps.is_available()
else:
    mps_available = False

def load_model(model_path, num_classes=11, device='cpu'):
    """Load a trained model for inference."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Load label mapping if available in the checkpoint
        label_to_idx = checkpoint.get('label_to_idx', None)
        
        # Create model instance
        model = MultiTaskNet(num_classes, model_type='efficientnet_b0')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Using device: {device}")
        
        if label_to_idx:
            print(f"Found label mapping with {len(label_to_idx)} classes")
            # Create reverse mapping for inference
            idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        else:
            print("No label mapping found in model checkpoint. Using hardcoded labels.")
            food_classes = [
                "apple", "avocado", "bagel", "biscuit", "blueberry_muffin", 
                "broccoli", "chicken_nugget", "cinnamon_roll", "corn", 
                "croissant", "strawberry"
            ]
            idx_to_label = {i: label for i, label in enumerate(food_classes)}
        
        return model, idx_to_label
    
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def prepare_image(image_path):
    """Prepare an image for inference."""
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        transformed_image = transform(image)
        
        return transformed_image.unsqueeze(0), image
    
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None

def run_inference(model, image_tensor, idx_to_label, device='cpu'):
    """Run inference on an image tensor."""
    try:
        # Move input to device
        image_tensor = image_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            class_logits, weight_pred = model(image_tensor)
            
            # Move tensors to CPU for post-processing (needed especially for MPS)
            class_logits = class_logits.to('cpu')
            weight_pred = weight_pred.to('cpu')
            
            # Get class prediction
            probabilities = torch.nn.functional.softmax(class_logits, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()
            
            # Get predicted weight
            predicted_weight = weight_pred.item()
            
            # Get predicted label
            predicted_label = idx_to_label.get(predicted_class_idx, f"Unknown_{predicted_class_idx}")
            
            # Get top 3 predictions
            top_values, top_indices = torch.topk(probabilities[0], min(3, len(idx_to_label)))
            top_predictions = [
                (idx_to_label.get(idx.item(), f"Unknown_{idx.item()}"), 
                 prob.item() * 100) 
                for idx, prob in zip(top_indices, top_values)
            ]
            
            return {
                'class_idx': predicted_class_idx,
                'class_name': predicted_label,
                'confidence': confidence * 100,
                'weight': predicted_weight,
                'top_predictions': top_predictions
            }
    
    except Exception as e:
        print(f"Error during inference: {e}")
        return None

def save_visualization(image, result, output_path):
    """Save visualization without showing the plot."""
    plt.figure(figsize=(10, 6))
    
    # Display image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Input Image')
    plt.axis('off')
    
    # Display predictions
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.title('Predictions')
    
    # Add predicted class and weight
    info_text = f"Predicted Food: {result['class_name']}\n"
    info_text += f"Confidence: {result['confidence']:.2f}%\n"
    info_text += f"Estimated Weight: {result['weight']:.2f}g\n\n"
    
    # Add top 3 predictions
    info_text += "Top Predictions:\n"
    for i, (label, prob) in enumerate(result['top_predictions']):
        info_text += f"{i+1}. {label}: {prob:.2f}%\n"
    
    plt.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Visualization saved to {output_path}")

def process_single_image(model, image_path, idx_to_label, device, output_dir):
    """Process a single image and save results."""
    print(f"Processing image: {image_path}")
    
    # Prepare image for inference
    image_tensor, original_image = prepare_image(image_path)
    
    if image_tensor is None:
        print(f"Skipping {image_path} due to processing error")
        return
    
    # Run inference
    result = run_inference(model, image_tensor, idx_to_label, device)
    
    if result is None:
        print(f"Inference failed for {image_path}")
        return
    
    # Print results
    print("\nInference Results:")
    print(f"Predicted Class: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print(f"Estimated Weight: {result['weight']:.2f}g")
    print("\nTop Predictions:")
    for i, (label, prob) in enumerate(result['top_predictions']):
        print(f"{i+1}. {label}: {prob:.2f}%")
    
    # Save visualization
    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"result_{base_name}")
    save_visualization(original_image, result, output_path)
    
    # Save result as JSON
    result_json = os.path.join(output_dir, f"result_{os.path.splitext(base_name)[0]}.json")
    with open(result_json, 'w') as f:
        json.dump(result, f, indent=4)

def process_limited_images(model, images_dir, idx_to_label, device, output_dir, limit=10):
    """Process a limited number of images from a directory."""
    if not os.path.exists(images_dir):
        print(f"Directory not found: {images_dir}")
        return
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(images_dir) if os.path.splitext(f.lower())[1] in image_extensions]
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return
    
    # Limit the number of images to process
    if limit > 0 and len(image_files) > limit:
        print(f"Found {len(image_files)} images, but limiting to {limit} images")
        image_files = image_files[:limit]
    else:
        print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(images_dir, img_file)
        print(f"\nProcessing image {i+1}/{len(image_files)}: {img_file}")
        process_single_image(model, img_path, idx_to_label, device, output_dir)
    
    print(f"\nAll results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Test the best_model.pth model with GPU support on macOS")
    parser.add_argument("--model_path", type=str, default="/Users/chalkiasantonios/Desktop/master-thesis/TRAIN_RESULTS/best_model.pth", 
                        help="Path to the trained model")
    parser.add_argument("--image_path", type=str, help="Path to a single food image for inference")
    parser.add_argument("--images_dir", type=str, help="Directory containing food images")
    parser.add_argument("--output_dir", type=str, default="/Users/chalkiasantonios/Desktop/master-thesis/results", 
                        help="Directory to save results")
    parser.add_argument("--num_classes", type=int, default=11, help="Number of food classes")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of images to process")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if GPU is available")
    
    args = parser.parse_args()
    
    # Check if either image_path or images_dir is provided
    if not args.image_path and not args.images_dir:
        print("Error: Either --image_path or --images_dir must be provided")
        parser.print_help()
        sys.exit(1)
    
    # Set device based on availability
    if args.cpu:
        device = 'cpu'
    else:
        if torch.cuda.is_available():
            device = 'cuda'
        elif mps_available:
            device = 'mps'  # Use MPS on macOS
        else:
            device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Load model
    model, idx_to_label = load_model(args.model_path, args.num_classes, device)
    
    # Process images
    if args.image_path:
        process_single_image(model, args.image_path, idx_to_label, device, args.output_dir)
    elif args.images_dir:
        process_limited_images(model, args.images_dir, idx_to_label, device, args.output_dir, args.limit)

if __name__ == "__main__":
    main()
