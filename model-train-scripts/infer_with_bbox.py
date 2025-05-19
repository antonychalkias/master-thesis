#!/usr/bin/env python3
"""
Enhanced visualization script for food recognition and weight estimation model.
This script adds bounding boxes around detected food items.
"""

import os
import sys
import torch
import argparse
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from model import MultiTaskNet

# From test_model.py
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
        original_size = image.size  # Store original size for bounding box scaling
        transformed_image = transform(image)
        
        return transformed_image.unsqueeze(0), image, original_size
    
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None, None

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

def visualize_with_bounding_box(image, result, output_path, bbox_method='center'):
    """
    Save visualization with bounding box around the detected food item.
    
    Args:
        image: PIL Image
        result: Inference result dictionary
        output_path: Path to save the visualization
        bbox_method: Method to use for creating bounding box
                     'center' - create box around center of image
                     'segmentation' - use simple segmentation to find food
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Convert PIL image to numpy array for matplotlib
    img_array = np.array(image)
    
    # Display original image with bounding box in first subplot
    ax1.imshow(img_array)
    ax1.set_title('Detected Food')
    ax1.axis('off')
    
    # Create bounding box
    if bbox_method == 'center':
        # Simple approach: Create a box at the center covering ~70% of the image
        h, w = img_array.shape[:2]
        box_w, box_h = int(w * 0.7), int(h * 0.7)
        x1, y1 = (w - box_w) // 2, (h - box_h) // 2
        x2, y2 = x1 + box_w, y1 + box_h
    else:  # Use simple segmentation
        # This is a simplified approach that works best with food on a contrasting background
        try:
            # Convert to grayscale and apply thresholding
            from skimage import filters, morphology, measure
            gray = np.mean(img_array, axis=2)
            thresh = filters.threshold_otsu(gray)
            binary = gray < thresh  # Dark object on light background
            
            # Clean up with morphological operations
            binary = morphology.remove_small_objects(binary, min_size=500)
            binary = morphology.remove_small_holes(binary, area_threshold=500)
            binary = morphology.binary_closing(binary, morphology.disk(10))
            
            # Find largest connected component
            labels = measure.label(binary)
            regions = measure.regionprops(labels)
            if regions:
                largest_region = max(regions, key=lambda r: r.area)
                y1, x1, y2, x2 = largest_region.bbox
            else:
                # Fallback to centered box
                h, w = img_array.shape[:2]
                box_w, box_h = int(w * 0.7), int(h * 0.7)
                x1, y1 = (w - box_w) // 2, (h - box_h) // 2
                x2, y2 = x1 + box_w, y1 + box_h
        except (ImportError, Exception) as e:
            print(f"Error in segmentation: {e}, falling back to centered box")
            # Fallback to centered box
            h, w = img_array.shape[:2]
            box_w, box_h = int(w * 0.7), int(h * 0.7)
            x1, y1 = (w - box_w) // 2, (h - box_h) // 2
            x2, y2 = x1 + box_w, y1 + box_h
    
    # Add rectangle patch to the image
    rect = patches.Rectangle(
        (x1, y1), x2-x1, y2-y1, 
        linewidth=3, edgecolor='lime', facecolor='none'
    )
    ax1.add_patch(rect)
    
    # Add text label with confidence on top of the bounding box
    label_text = f"{result['class_name']} ({result['confidence']:.1f}%)"
    ax1.text(
        x1, y1-10, label_text,
        color='white', fontsize=12, weight='bold',
        bbox=dict(facecolor='green', alpha=0.7, edgecolor='none', pad=2)
    )
    
    # Display predictions in second subplot
    ax2.axis('off')
    ax2.set_title('Predictions')
    
    # Add predicted class and weight
    info_text = f"Predicted Food: {result['class_name']}\n"
    info_text += f"Confidence: {result['confidence']:.2f}%\n"
    info_text += f"Estimated Weight: {result['weight']:.2f}g\n\n"
    
    # Add top 3 predictions
    info_text += "Top Predictions:\n"
    for i, (label, prob) in enumerate(result['top_predictions']):
        info_text += f"{i+1}. {label}: {prob:.2f}%\n"
    
    ax2.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save the visualization
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Visualization with bounding box saved to {output_path}")
    
    # Also save a combined visualization as a single image using PIL for better quality
    create_combined_visualization(image, result, x1, y1, x2, y2, 
                                 output_path.replace('.jpg', '_combined.jpg'))

def create_combined_visualization(image, result, x1, y1, x2, y2, output_path):
    """Create a high-quality combined visualization with PIL"""
    # Make a copy of the image to draw on
    img_with_box = image.copy()
    draw = ImageDraw.Draw(img_with_box)
    
    # Draw bounding box
    draw.rectangle([x1, y1, x2, y2], outline="lime", width=4)
    
    # Try to load a font, use default if not available
    try:
        # Try to use a system font
        font = ImageFont.truetype("Arial", 24)
    except IOError:
        # Use default font if Arial is not available
        font = ImageFont.load_default()
    
    # Draw label
    label_text = f"{result['class_name']} ({result['confidence']:.1f}%)"
    text_bbox = draw.textbbox((0, 0), label_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Draw text background
    draw.rectangle([x1, y1-text_height-10, x1+text_width+10, y1-5], fill="green")
    
    # Draw text
    draw.text((x1+5, y1-text_height-7), label_text, fill="white", font=font)
    
    # Save the enhanced image
    img_with_box.save(output_path)
    print(f"Combined visualization saved to {output_path}")

def process_image_with_bbox(model, image_path, idx_to_label, device, output_dir, bbox_method='center'):
    """Process a single image and save results with bounding box."""
    print(f"Processing image: {image_path}")
    
    # Prepare image for inference
    image_tensor, original_image, original_size = prepare_image(image_path)
    
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
    
    # Save visualization with bounding box
    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"result_{base_name}")
    visualize_with_bounding_box(original_image, result, output_path, bbox_method)
    
    # Save result as JSON
    result_json = os.path.join(output_dir, f"result_{os.path.splitext(base_name)[0]}.json")
    with open(result_json, 'w') as f:
        json.dump(result, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Run inference with bounding boxes on food images")
    parser.add_argument("--model_path", type=str, 
                        default="/Users/chalkiasantonios/Desktop/master-thesis/TRAIN_RESULTS/best_model.pth",
                        help="Path to the trained model")
    parser.add_argument("--image_path", type=str, help="Path to a single food image")
    parser.add_argument("--images_dir", type=str, help="Directory containing food images")
    parser.add_argument("--output_dir", type=str, default="/Users/chalkiasantonios/Desktop/master-thesis/results_bbox",
                        help="Directory to save results")
    parser.add_argument("--num_classes", type=int, default=11, help="Number of food classes")
    parser.add_argument("--limit", type=int, default=5, help="Limit number of images to process")
    parser.add_argument("--bbox_method", type=str, choices=['center', 'segmentation'], default='center',
                        help="Method for creating bounding boxes")
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
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'  # Use MPS on macOS
        else:
            device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, idx_to_label = load_model(args.model_path, args.num_classes, device)
    
    # Process images
    if args.image_path:
        process_image_with_bbox(model, args.image_path, idx_to_label, device, args.output_dir, args.bbox_method)
    elif args.images_dir:
        # Get all image files in the directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in os.listdir(args.images_dir) if os.path.splitext(f.lower())[1] in image_extensions]
        
        if not image_files:
            print(f"No images found in {args.images_dir}")
            return
        
        # Limit the number of images to process if specified
        if args.limit > 0 and len(image_files) > args.limit:
            print(f"Found {len(image_files)} images, but limiting to {args.limit} images")
            image_files = image_files[:args.limit]
        else:
            print(f"Found {len(image_files)} images to process")
        
        # Process each image
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(args.images_dir, img_file)
            print(f"\nProcessing image {i+1}/{len(image_files)}: {img_file}")
            process_image_with_bbox(model, img_path, idx_to_label, device, args.output_dir, args.bbox_method)
        
        print(f"\nAll results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
