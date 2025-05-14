#!/usr/bin/env python3
"""
Inference script for food recognition and weight estimation model.
"""

import argparse
import os
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
    print("Warning: scikit-image or scipy not found. Advanced food detection will not be available.")
    HAS_SKIMAGE = False

def parse_args():
    parser = argparse.ArgumentParser(description="Food recognition and weight estimation inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--image_path", type=str, default=None, help="Path to a single food image")
    parser.add_argument("--images_dir", type=str, default=None, help="Directory with multiple food images")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    return parser.parse_args()

def load_model(model_path):
    """Load the trained model and label mapping."""
    # Handle new PyTorch serialization behavior in newer versions
    try:
        # First try with weights_only=False (old behavior)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    except (RuntimeError, TypeError) as e:
        print(f"Warning: {e}")
        print("Trying alternative loading method...")
        import numpy as np
        # Add numpy scalar to safe globals
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    label_to_idx = checkpoint.get('label_to_idx', {})
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    
    num_classes = len(label_to_idx)
    model = MultiTaskNet(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, idx_to_label

def process_image(image_path, model, idx_to_label, device):
    """Process a single image and return the predictions."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    original_image = image.copy()
    
    # Prepare image for model
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        class_logits, weight_pred = model(input_tensor)
    
    # Process classification results
    probabilities = torch.nn.functional.softmax(class_logits, dim=1)
    top_prob, top_class = torch.topk(probabilities, k=3, dim=1)
    
    # Convert to numpy for easier handling
    top_classes = top_class.squeeze().cpu().numpy()
    top_probs = top_prob.squeeze().cpu().numpy()
    
    # Get the food labels and probabilities
    food_predictions = [
        {
            "label": idx_to_label[idx], 
            "probability": float(prob)
        }
        for idx, prob in zip(top_classes, top_probs)
    ]
    
    # Get weight prediction
    predicted_weight = float(weight_pred.item())
    
    results = {
        "food_predictions": food_predictions,
        "predicted_weight": predicted_weight,
        "image_path": image_path
    }
    
    return results, original_image

def find_food_region(image):
    """
    Attempt to find the food region in the image using more advanced image processing.
    Returns bounding box coordinates (x1, y1, x2, y2) for the food.
    """
    # Convert PIL Image to numpy array for processing
    img_array = np.array(image)
    width, height = image.size
    
    # Default fallback values - will be used if detection fails
    # Using a smaller default box (60% of image instead of 80%)
    default_x_min = (width - int(width * 0.6)) // 2
    default_y_min = (height - int(height * 0.6)) // 2
    default_x_max = default_x_min + int(width * 0.6)
    default_y_max = default_y_min + int(height * 0.6)
    
    # Check if required libraries are available
    if not HAS_SKIMAGE:
        print("Advanced food detection unavailable: scikit-image not installed")
        return default_x_min, default_y_min, default_x_max, default_y_max
    
    try:
        # Convert to grayscale if image is color
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            # Convert to LAB color space which is better for color segmentation
            lab_img = color.rgb2lab(img_array[:,:,:3] / 255.0)
            
            # Apply SLIC superpixel segmentation with more segments for finer detection
            # This groups similar color regions together
            segments = segmentation.slic(lab_img, n_segments=150, compactness=15, start_label=1)
            
            # Find the segment closest to the center (likely to be the food)
            center_y, center_x = int(height / 2), int(width / 2)
            center_segment = segments[center_y, center_x]
            
            # Create a mask of the center segment and surrounding similar segments
            mask = np.zeros_like(segments, dtype=bool)
            mask[segments == center_segment] = True
            
            # Analyze the surrounding segments and include them if they're similar
            props = measure.regionprops(segments)
            
            # Expand to include similar nearby segments but with smaller radius
            for prop in props:
                prop_center = prop.centroid
                distance = np.sqrt((prop_center[0] - center_y)**2 + (prop_center[1] - center_x)**2)
                if distance < min(height, width) * 0.3:  # Reduced from 0.4 to 0.3 for tighter boxing
                    mask[segments == prop.label] = True
            
            # Use morphological operations to clean up the mask
            mask = ndimage.binary_dilation(mask, iterations=3)
            mask = ndimage.binary_erosion(mask, iterations=2)
            mask = ndimage.binary_dilation(mask, iterations=2)
            
            # Find contours of the mask
            contours = measure.find_contours(mask.astype(float), 0.5)
            
            if contours and len(contours) > 0:
                # Find the largest contour by area
                largest_contour = max(contours, key=lambda x: len(x))
                
                # Get bounding box from contour
                y_indices, x_indices = largest_contour[:, 0], largest_contour[:, 1]
                
                # Check if we have enough points to form a meaningful bounding box
                if len(x_indices) < 10 or len(y_indices) < 10:
                    raise ValueError("Contour too small, falling back to edge detection")
            else:
                # Fallback to edge detection if no contours found
                raise ValueError("No contours found, falling back to edge detection")
        
        # Fallback to edge detection
        else:
            raise ValueError("Image format requires edge detection fallback")
            
    except Exception as e:
        print(f"Color-based segmentation failed: {e}. Trying edge detection...")
        try:
            # Use edge detection to find the food region
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                gray_img = np.dot(img_array[:, :, :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            else:
                gray_img = img_array
            
            # Apply lighter Gaussian blur to preserve more edge details
            blurred = filters.gaussian(gray_img, sigma=0.8)
            
            # Use Canny edge detection with lower sigma for more precise edge detection
            edges = feature.canny(blurred, sigma=1.5)
            
            # Find the coordinates of the edges
            y_indices, x_indices = np.where(edges)
            
            # Check if we found any edges
            if len(x_indices) < 10 or len(y_indices) < 10:
                raise ValueError("Insufficient edges detected")
                
        except Exception as e2:
            print(f"Edge detection also failed: {e2}. Using default bounding box.")
            return default_x_min, default_y_min, default_x_max, default_y_max
    
    try:
        # Get bounding box coordinates
        x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
        y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
        
        # Check if the bounding box is too small (less than 15% of the image)
        min_bbox_size = min(width, height) * 0.15
        if (x_max - x_min < min_bbox_size) or (y_max - y_min < min_bbox_size):
            print("Detected region too small, using default bounding box")
            return default_x_min, default_y_min, default_x_max, default_y_max
        
        # Calculate the food object size relative to the image
        box_width = x_max - x_min
        box_height = y_max - y_min
        size_ratio = (box_width * box_height) / (width * height)
        
        print(f"Detected food region size: {box_width}x{box_height} pixels")
        print(f"Size ratio (food area / image area): {size_ratio:.3f}")
        
        # Adaptive padding: smaller padding for larger food items
        # This ensures small foods get more padding and large foods get less padding
        if size_ratio > 0.3:  # Large food item
            padding_factor = 0.05  # 5% padding
            print("Large food item detected, using 5% padding")
        elif size_ratio > 0.15:  # Medium food item
            padding_factor = 0.08  # 8% padding
            print("Medium food item detected, using 8% padding")
        else:  # Small food item
            padding_factor = 0.12  # 12% padding
            print("Small food item detected, using 12% padding")
            
        padding_x = int(width * padding_factor)
        padding_y = int(height * padding_factor)
        print(f"Applying padding: {padding_x}x{padding_y} pixels")
        
        x_min = max(0, x_min - padding_x)
        y_min = max(0, y_min - padding_y)
        x_max = min(width, x_max + padding_x)
        y_max = min(height, y_max + padding_y)
        
        return x_min, y_min, x_max, y_max
        
    except Exception as e:
        print(f"Error processing detection results: {e}. Using default bounding box.")
        return default_x_min, default_y_min, default_x_max, default_y_max

def visualize_results(image, results, output_path):
    """Create and save a visualization of the results."""
    # Create a figure
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis('off')
    
    # Get image dimensions
    width, height = image.size
    
    # Try to find food region
    x_min, y_min, x_max, y_max = find_food_region(image)
    
    # Draw the bounding box with the top prediction's color based on confidence
    top_probability = results['food_predictions'][0]['probability']
    # Color transitions from red (low confidence) to green (high confidence)
    box_color = (1 - top_probability, top_probability, 0)
    
    # Create a Rectangle patch
    bbox = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                   fill=False, edgecolor=box_color, linewidth=3, alpha=0.7)
    plt.gca().add_patch(bbox)
    
    # Add text overlay with predictions
    prediction_text = f"Food: {results['food_predictions'][0]['label']} ({results['food_predictions'][0]['probability']:.2f})\n"
    prediction_text += f"Weight: {results['predicted_weight']:.1f}g"
    
    plt.text(10, 30, prediction_text, color='white', fontsize=12, 
             bbox=dict(facecolor='black', alpha=0.7))
    
    # Add confidence information
    confidence_text = "Top 3 predictions:"
    for idx, pred in enumerate(results['food_predictions']):
        confidence_text += f"\n{idx+1}. {pred['label']}: {pred['probability']:.2f}"
        
    plt.text(10, height - 80, confidence_text, color='white', fontsize=10,
             bbox=dict(facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Results saved to {output_path}")

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, idx_to_label = load_model(args.model_path)
    model.to(device)
    
    if args.image_path:
        # Process a single image
        image_name = os.path.basename(args.image_path)
        results, original_image = process_image(args.image_path, model, idx_to_label, device)
        
        # Save results
        output_path = os.path.join(args.output_dir, f"{os.path.splitext(image_name)[0]}_result.jpg")
        visualize_results(original_image, results, output_path)
        
        # Save JSON with detailed results
        json_path = os.path.join(args.output_dir, f"{os.path.splitext(image_name)[0]}_result.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
            
    elif args.images_dir:
        # Process all images in the directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files = [f for f in os.listdir(args.images_dir) 
                      if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        print(f"Found {len(image_files)} images to process")
        
        for image_file in image_files:
            image_path = os.path.join(args.images_dir, image_file)
            results, original_image = process_image(image_path, model, idx_to_label, device)
            
            # Save results
            output_path = os.path.join(args.output_dir, f"{os.path.splitext(image_file)[0]}_result.jpg")
            visualize_results(original_image, results, output_path)
            
            # Save JSON with detailed results
            json_path = os.path.join(args.output_dir, f"{os.path.splitext(image_file)[0]}_result.json")
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=4)
            
            print(f"Processed {image_file}")
    else:
        print("Please provide either --image_path or --images_dir")

if __name__ == "__main__":
    main()
