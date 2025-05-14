#!/usr/bin/env python3
"""
Inference script for food recognition and weight estimation model.
"""

import argparse
import os
import sys

# Add the parent directory to the Python path so we can import the model module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
from model import MultiTaskNet  # Import the model architecture from model.py

# Import additional libraries for image processing
try:
    from skimage import color, segmentation, measure, feature, filters
    from scipy import ndimage
    from matplotlib.patches import Rectangle
    HAS_SKIMAGE = True
except ImportError:
    print("Warning: scikit-image not found. Advanced visualization features will be disabled.")
    HAS_SKIMAGE = False

def parse_args():
    """
    Parse command-line arguments for the inference script.
    
    Returns:
        args: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run inference with food recognition and weight estimation model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--image_path", type=str, default=None, help="Path to a single image")
    parser.add_argument("--images_dir", type=str, default=None, help="Path to a directory with images")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save results")
    parser.add_argument("--show_results", action="store_true", help="Display results")
    parser.add_argument("--advanced_viz", action="store_true", help="Use advanced visualization techniques")
    return parser.parse_args()

def get_device():
    """
    Get the best available device for inference (CUDA, MPS, or CPU).
    
    Returns:
        device: PyTorch device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_model(model_path, device):
    """
    Load the trained model from a checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        device: PyTorch device
    
    Returns:
        model: Loaded model
        idx_to_label: Mapping from index to label name
    """
    print(f"Loading model from {model_path}")
    
    # Add numpy.core.multiarray.scalar to the safe globals list for PyTorch 2.6+ compatibility
    torch.serialization.add_safe_globals(['numpy.core.multiarray.scalar'])
    
    try:
        # First try to load as a new format checkpoint (dictionary)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        label_to_idx = checkpoint.get('label_to_idx', {})
        
        # Check if model_state_dict is available
        if 'model_state_dict' in checkpoint:
            # Initialize model with number of classes
            num_classes = len(label_to_idx) if label_to_idx else 11  # Default to 11 classes
            model = MultiTaskNet(num_classes=num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Try to load as direct state dict
            num_classes = 11  # Default to 11 classes for older models
            model = MultiTaskNet(num_classes=num_classes)
            model.load_state_dict(checkpoint)
            
    except Exception as e:
        print(f"Error loading checkpoint as dictionary: {e}")
        print("Attempting to load as direct state dict...")
        
        # Try to load as a direct state dict
        num_classes = 11  # Default to 11 classes for older models
        model = MultiTaskNet(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        label_to_idx = {}  # No label mapping available
    
    # Create reverse mapping
    idx_to_label = {v: k for k, v in label_to_idx.items()} if label_to_idx else {}
    
    # If no labels were loaded, use default labels
    if not idx_to_label:
        print("Warning: No label mapping found in model checkpoint. Using default labels.")
        default_labels = ["bagel", "donut", "hamburger", "hot dog", "pizza", "sandwich", 
                         "taco", "burrito", "breakfast sandwich", "salad", "other"]
        idx_to_label = {i: label for i, label in enumerate(default_labels)}
    
    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()
    
    return model, idx_to_label

def preprocess_image(image_path):
    """
    Load and preprocess an image for inference.
    
    Args:
        image_path: Path to the image
    
    Returns:
        image_tensor: Preprocessed image tensor
        original_image: Original image for visualization
    """
    # Define preprocessing transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load image
    original_image = Image.open(image_path).convert("RGB")
    
    # Preprocess image
    image_tensor = transform(original_image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    return image_tensor, original_image

def process_image(image_path, model, idx_to_label, device, advanced_viz=False):
    """
    Process a single image and get food recognition and weight estimation results.
    
    Args:
        image_path: Path to the image
        model: Loaded model
        idx_to_label: Mapping from index to label name
        device: PyTorch device
        advanced_viz: Whether to use advanced visualization techniques
    
    Returns:
        results: Dictionary with inference results
        visualization: Visualization image
    """
    # Load and preprocess image
    image_tensor, original_image = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        outputs_class, outputs_weight = model(image_tensor)
        
        # Get top-3 predictions
        probabilities = torch.nn.functional.softmax(outputs_class, dim=1)[0]
        top3_probs, top3_indices = torch.topk(probabilities, 3)
        top3_probs = top3_probs.cpu().numpy()
        top3_indices = top3_indices.cpu().numpy()
        
        # Get weight prediction
        weight_pred = outputs_weight.item()
    
    # Create visualization
    if advanced_viz and HAS_SKIMAGE:
        visualization = advanced_visualization(original_image, top3_indices, top3_probs, weight_pred, idx_to_label)
    else:
        visualization = basic_visualization(original_image, top3_indices, top3_probs, weight_pred, idx_to_label)
    
    # Prepare results
    results = {
        "predictions": [
            {
                "label": idx_to_label.get(idx, f"Class {idx}"),
                "probability": float(prob)
            } for idx, prob in zip(top3_indices, top3_probs)
        ],
        "weight_prediction": weight_pred,
        "image_path": image_path,
        "image_name": os.path.basename(image_path)
    }
    
    return results, visualization

def basic_visualization(image, top3_indices, top3_probs, weight_pred, idx_to_label):
    """
    Create a basic visualization of inference results.
    
    Args:
        image: Original image
        top3_indices: Top 3 predicted class indices
        top3_probs: Top 3 predicted class probabilities
        weight_pred: Predicted weight
        idx_to_label: Mapping from index to label name
    
    Returns:
        fig: Matplotlib figure with visualization
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display image
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title("Input Image")
    
    # Display predictions
    predictions = [f"{idx_to_label.get(idx, f'Class {idx}')} ({prob:.2%})" for idx, prob in zip(top3_indices, top3_probs)]
    y_pos = np.arange(len(predictions))
    ax2.barh(y_pos, top3_probs, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(predictions)
    ax2.invert_yaxis()
    ax2.set_xlabel('Probability')
    ax2.set_title(f"Top Predictions (Weight: {weight_pred:.1f}g)")
    
    plt.tight_layout()
    return fig

def advanced_visualization(image, top3_indices, top3_probs, weight_pred, idx_to_label):
    """
    Create an advanced visualization of inference results using segmentation.
    
    Args:
        image: Original image
        top3_indices: Top 3 predicted class indices
        top3_probs: Top 3 predicted class probabilities
        weight_pred: Predicted weight
        idx_to_label: Mapping from index to label name
    
    Returns:
        fig: Matplotlib figure with visualization
    """
    # Convert PIL image to numpy array
    img_np = np.array(image)
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Display original image
    ax1.imshow(img_np)
    ax1.axis('off')
    ax1.set_title("Original Image")
    
    # Try to segment the food item
    try:
        # Convert to grayscale and apply adaptive thresholding
        gray = color.rgb2gray(img_np)
        thresh = filters.threshold_otsu(gray)
        binary = gray > thresh
        
        # Clean up binary image
        cleaned = ndimage.binary_closing(binary, iterations=2)
        cleaned = ndimage.binary_opening(cleaned, iterations=2)
        
        # Find contours
        contours = measure.find_contours(cleaned, 0.8)
        
        # Display segmentation
        ax2.imshow(img_np)
        for contour in contours:
            if len(contour) > 100:  # Filter small contours
                ax2.plot(contour[:, 1], contour[:, 0], linewidth=2)
        
        ax2.axis('off')
        ax2.set_title("Food Segmentation")
    except Exception as e:
        # If segmentation fails, just show the original image
        print(f"Warning: Segmentation failed: {e}")
        ax2.imshow(img_np)
        ax2.axis('off')
        ax2.set_title("Segmentation Failed")
    
    # Display predictions
    top_prediction = idx_to_label.get(top3_indices[0], f"Class {top3_indices[0]}")
    ax3.set_title(f"Predictions: {top_prediction} ({top3_probs[0]:.2%})\nWeight: {weight_pred:.1f}g")
    ax3.axis('off')
    
    # Bar chart for predictions
    predictions = [idx_to_label.get(idx, f"Class {idx}") for idx in top3_indices]
    y_pos = np.arange(len(predictions))
    ax3.barh(y_pos, top3_probs, align='center')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(predictions)
    ax3.set_xlim(0, 1)
    for i, prob in enumerate(top3_probs):
        ax3.text(prob + 0.01, i, f"{prob:.2%}", va='center')
    
    plt.tight_layout()
    return fig

def main():
    args = parse_args()
    
    # Check if at least one of image_path or images_dir is specified
    if not args.image_path and not args.images_dir:
        print("Error: Either --image_path or --images_dir must be specified")
        return
    
    # Get device
    device = get_device()
    
    # Load model
    model, idx_to_label = load_model(args.model_path, device)
    
    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Process single image
    if args.image_path:
        results, visualization = process_image(args.image_path, model, idx_to_label, device, args.advanced_viz)
        
        # Save visualization
        if args.output_dir:
            output_path = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.image_path))[0]}_result.png")
            visualization.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {output_path}")
        
        # Display results
        if args.show_results:
            plt.show()
        else:
            plt.close(visualization)
        
        # Print results
        print("\nInference Results:")
        print(f"Image: {os.path.basename(args.image_path)}")
        print(f"Predicted Weight: {results['weight_prediction']:.1f}g")
        print("Top Predictions:")
        for i, pred in enumerate(results['predictions']):
            print(f"  {i+1}. {pred['label']} ({pred['probability']:.2%})")
    
    # Process directory of images
    else:
        all_results = []
        image_paths = [os.path.join(args.images_dir, f) for f in os.listdir(args.images_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for image_path in image_paths:
            print(f"Processing {os.path.basename(image_path)}...")
            results, visualization = process_image(image_path, model, idx_to_label, device, args.advanced_viz)
            all_results.append(results)
            
            # Save visualization
            if args.output_dir:
                output_path = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_result.png")
                visualization.savefig(output_path, dpi=300, bbox_inches='tight')
            
            # Close figure to free memory
            plt.close(visualization)
        
        # Save all results to JSON
        if args.output_dir:
            json_path = os.path.join(args.output_dir, "all_results.json")
            with open(json_path, 'w') as f:
                json.dump(all_results, f, indent=4)
            print(f"Saved all results to {json_path}")
        
        # Print summary
        print("\nInference Summary:")
        print(f"Processed {len(image_paths)} images")
        print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
