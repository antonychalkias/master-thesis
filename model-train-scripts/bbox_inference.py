#!/usr/bin/env python3
"""
Food Recognition with Bounding Boxes using PIL.
This script performs inference and adds bounding boxes using only PIL.
"""

import os
import sys
import torch
import argparse
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from torchvision import transforms
from model import MultiTaskNet

def load_model(model_path, num_classes=11, device='cpu'):
    """Load trained model for inference."""
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

def add_bounding_box(image, result, output_path):
    """Add a bounding box to the image using PIL."""
    try:
        # Make a copy of the image
        img_with_box = image.copy()
        draw = ImageDraw.Draw(img_with_box)
        
        # Get image dimensions
        width, height = image.size
        
        # Create a centered bounding box (70% of image)
        box_w, box_h = int(width * 0.7), int(height * 0.7)
        x1, y1 = (width - box_w) // 2, (height - box_h) // 2
        x2, y2 = x1 + box_w, y1 + box_h
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=4)
        
        # Try to load a font, use default if not available
        try:
            # Try to use a system font
            font = ImageFont.truetype("Arial", 20)
        except IOError:
            try:
                # Try another common font
                font = ImageFont.truetype("DejaVuSans.ttf", 20)
            except IOError:
                # Use default font if others are not available
                font = ImageFont.load_default()
        
        # Prepare label text
        label_text = f"{result['class_name']} ({result['confidence']:.1f}%)"
        
        # Measure text size
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Draw text background
        draw.rectangle([x1, y1-text_height-4, x1+text_width+8, y1], fill=(0, 128, 0))
        
        # Draw text
        draw.text((x1+4, y1-text_height-2), label_text, fill=(255, 255, 255), font=font)
        
        # Also add weight info at the bottom
        weight_text = f"Est. Weight: {result['weight']:.1f}g"
        weight_bbox = draw.textbbox((0, 0), weight_text, font=font)
        weight_width = weight_bbox[2] - weight_bbox[0]
        
        # Draw weight text background and text
        draw.rectangle([x1, y2, x1+weight_width+8, y2+text_height+4], fill=(0, 128, 0))
        draw.text((x1+4, y2+2), weight_text, fill=(255, 255, 255), font=font)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Save the image
        img_with_box.save(output_path)
        print(f"Image with bounding box saved to {output_path}")
        
        # Create a side-by-side image with the original and the prediction
        combined_width = width * 2 + 20  # 20px gap between images
        combined_height = height + 100   # Extra space for text at bottom
        combined_img = Image.new('RGB', (combined_width, combined_height), color=(255, 255, 255))
        
        # Paste original image on the left
        combined_img.paste(image, (0, 0))
        
        # Paste boxed image on the right
        combined_img.paste(img_with_box, (width + 20, 0))
        
        # Add title and prediction text
        combined_draw = ImageDraw.Draw(combined_img)
        
        # Add titles
        combined_draw.text((width//2 - 50, height + 10), "Original Image", fill=(0, 0, 0), font=font)
        combined_draw.text((width + width//2 - 50, height + 10), "Prediction", fill=(0, 0, 0), font=font)
        
        # Add prediction details at the bottom
        pred_text = f"Top predictions: "
        for i, (label, prob) in enumerate(result['top_predictions']):
            pred_text += f"{label} ({prob:.1f}%), "
        
        pred_text = pred_text[:-2]  # Remove trailing comma and space
        combined_draw.text((10, height + 50), pred_text, fill=(0, 0, 0), font=font)
        
        # Save the combined image
        combined_output = output_path.replace('.jpg', '_combined.jpg')
        combined_img.save(combined_output)
        print(f"Combined visualization saved to {combined_output}")
        
        return True
    
    except Exception as e:
        print(f"Error adding bounding box: {e}")
        return False

def process_image(model, image_path, idx_to_label, device, output_dir):
    """Process a single image and add bounding box."""
    try:
        print(f"Processing image: {image_path}")
        
        # Prepare image for inference
        image_tensor, original_image = prepare_image(image_path)
        
        if image_tensor is None:
            print(f"Skipping {image_path} due to processing error")
            return False
        
        # Run inference
        result = run_inference(model, image_tensor, idx_to_label, device)
        
        if result is None:
            print(f"Inference failed for {image_path}")
            return False
        
        # Print results
        print("\nInference Results:")
        print(f"Predicted Class: {result['class_name']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"Estimated Weight: {result['weight']:.2f}g")
        print("\nTop Predictions:")
        for i, (label, prob) in enumerate(result['top_predictions']):
            print(f"{i+1}. {label}: {prob:.2f}%")
        
        # Save the image with bounding box
        base_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"bbox_{base_name}")
        
        # Add bounding box
        success = add_bounding_box(original_image, result, output_path)
        
        # Save result as JSON
        result_json = os.path.join(output_dir, f"result_{os.path.splitext(base_name)[0]}.json")
        with open(result_json, 'w') as f:
            json.dump(result, f, indent=4)
        
        return success
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Food recognition with bounding boxes")
    parser.add_argument("--model_path", type=str, 
                        default="/Users/chalkiasantonios/Desktop/master-thesis/TRAIN_RESULTS/best_model.pth", 
                        help="Path to the model")
    parser.add_argument("--image_path", type=str, help="Path to a single image")
    parser.add_argument("--images_dir", type=str, help="Directory with multiple images")
    parser.add_argument("--output_dir", type=str, default="/Users/chalkiasantonios/Desktop/master-thesis/results_bbox",
                        help="Output directory")
    parser.add_argument("--num_classes", type=int, default=11, help="Number of classes")
    parser.add_argument("--limit", type=int, default=5, help="Limit number of images to process")
    
    args = parser.parse_args()
    
    # Check that either image_path or images_dir is provided
    if not args.image_path and not args.images_dir:
        parser.error("Either --image_path or --images_dir must be specified")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine device to use
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'  # Apple Silicon GPU
    
    print(f"Using device: {device}")
    
    # Load model
    try:
        model, idx_to_label = load_model(args.model_path, args.num_classes, device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
    
    success_count = 0
    fail_count = 0
    
    # Process single image or directory
    if args.image_path:
        if process_image(model, args.image_path, idx_to_label, device, args.output_dir):
            success_count += 1
        else:
            fail_count += 1
    
    elif args.images_dir:
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = [f for f in os.listdir(args.images_dir) 
                      if os.path.splitext(f.lower())[1] in image_extensions]
        
        if not image_files:
            print(f"No images found in {args.images_dir}")
            sys.exit(1)
        
        # Limit number of images if needed
        if args.limit > 0 and len(image_files) > args.limit:
            print(f"Found {len(image_files)} images, limiting to {args.limit}")
            image_files = image_files[:args.limit]
        else:
            print(f"Processing {len(image_files)} images")
        
        # Process each image
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(args.images_dir, img_file)
            print(f"Processing image {i+1}/{len(image_files)}: {img_file}")
            
            if process_image(model, img_path, idx_to_label, device, args.output_dir):
                success_count += 1
            else:
                fail_count += 1
    
    # Print summary
    print("\nProcessing complete!")
    print(f"Successfully processed: {success_count} images")
    if fail_count > 0:
        print(f"Failed to process: {fail_count} images")
    
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
