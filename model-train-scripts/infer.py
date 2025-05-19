#!/usr/bin/env python3
"""
Inference script for food recognition and weight estimation model.
This script loads a trained model and performs inference on a single image
or directory of images.
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
    torch.backends.mps.is_available()  # Initialize MPS

def load_model(model_path, num_classes=11, device='cpu'):
    """
    Load a trained model for inference.
    
    Args:
        model_path (str): Path to the saved model
        num_classes (int): Number of food classes
        device (str): Device to load the model to ('cpu', 'cuda', or 'mps')
        
    Returns:
        tuple: (model, label_to_idx) - The loaded model and label-to-index mapping
    """
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
            print("No label mapping found in model checkpoint. Using hardcoded labels from training_results.json")
            # Hardcoded labels from training_results.json
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
    """
    Prepare an image for inference by applying appropriate transformations.
    
    Args:
        image_path (str): Path to the image
        
    Returns:
        torch.Tensor: Processed image tensor ready for model inference
    """
    try:
        # Define inference transforms (same as validation transforms)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        transformed_image = transform(image)
        
        # Add batch dimension
        return transformed_image.unsqueeze(0), image
    
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None

def run_inference(model, image_tensor, idx_to_label, device='cpu'):
    """
    Run inference on an image tensor using the loaded model.
    
    Args:
        model: Trained PyTorch model
        image_tensor: Preprocessed image tensor
        idx_to_label: Mapping from class indices to labels
        device: Device to run inference on
        
    Returns:
        dict: Inference results including class prediction and weight estimation
    """
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
                'confidence': confidence * 100,  # Convert to percentage
                'weight': predicted_weight,
                'top_predictions': top_predictions
            }
    
    except Exception as e:
        print(f"Error during inference: {e}")
        return None

def visualize_result(image, result, output_path=None):
    """
    Visualize the inference result with the image and predictions.
    
    Args:
        image: PIL Image
        result: Inference result dictionary
        output_path: Path to save the visualization (optional)
    """
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
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
        
        # Don't show the plot when saving to a file
        plt.close()
    else:
        plt.show()

def process_single_image(model, image_path, idx_to_label, device, output_dir=None):
    """
    Process a single image and display/save results
    
    Args:
        model: Trained model
        image_path: Path to the image
        idx_to_label: Mapping from indices to class labels
        device: Device to run inference on
        output_dir: Directory to save results (optional)
    """
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
    
    # Save/display visualization
    if output_dir:
        base_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"result_{base_name}")
        visualize_result(original_image, result, output_path)
        
        # Save result as JSON
        result_json = os.path.join(output_dir, f"result_{os.path.splitext(base_name)[0]}.json")
        with open(result_json, 'w') as f:
            json.dump(result, f, indent=4)
    else:
        visualize_result(original_image, result)

def process_directory(model, images_dir, idx_to_label, device, output_dir):
    """
    Process all images in a directory
    
    Args:
        model: Trained model
        images_dir: Directory containing images
        idx_to_label: Mapping from indices to class labels
        device: Device to run inference on
        output_dir: Directory to save results
    """
    if not os.path.exists(images_dir):
        print(f"Directory not found: {images_dir}")
        return
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Set matplotlib to non-interactive mode when processing directories
    plt.ioff()
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(images_dir) if os.path.splitext(f.lower())[1] in image_extensions]
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return
    
    print(f"Found {len(image_files)} images in {images_dir}")
    
    # Process each image
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(images_dir, img_file)
        print(f"\nProcessing image {i+1}/{len(image_files)}: {img_file}")
        process_single_image(model, img_path, idx_to_label, device, output_dir)
        
        # Stop after 10 images if there are too many (to avoid processing all 500+ images)
        if i >= 9:
            print(f"\nProcessed 10 images. To process more, specify individual images with --image_path")
            break
    
    # Reset matplotlib to interactive mode
    plt.ion()
    
    print(f"\nAll results saved to {output_dir}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run inference with food recognition and weight estimation model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--image_path", type=str, help="Path to a single food image for inference")
    parser.add_argument("--images_dir", type=str, help="Directory containing multiple food images")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--num_classes", type=int, default=11, help="Number of food classes")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for inference if available (CUDA on NVIDIA or MPS on Apple Silicon)")
    
    args = parser.parse_args()
    
    # Check if either image_path or images_dir is provided
    if not args.image_path and not args.images_dir:
        print("Error: Either --image_path or --images_dir must be provided")
        parser.print_help()
        sys.exit(1)
    
    # Set device
    if args.use_gpu:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'  # Use Metal Performance Shaders on macOS with Apple Silicon
        else:
            device = 'cpu'
    else:
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Load model
    model, idx_to_label = load_model(args.model_path, args.num_classes, device)
    
    # Process images
    if args.image_path:
        process_single_image(model, args.image_path, idx_to_label, device, args.output_dir)
    elif args.images_dir:
        process_directory(model, args.images_dir, idx_to_label, device, args.output_dir)
    
if __name__ == "__main__":
    main()
