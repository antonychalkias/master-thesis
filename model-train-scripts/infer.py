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

def parse_args():
    parser = argparse.ArgumentParser(description="Food recognition and weight estimation inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--image_path", type=str, default=None, help="Path to a single food image")
    parser.add_argument("--images_dir", type=str, default=None, help="Directory with multiple food images")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    return parser.parse_args()

def load_model(model_path):
    """Load the trained model and label mapping."""
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

def visualize_results(image, results, output_path):
    """Create and save a visualization of the results."""
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis('off')
    
    # Add text overlay with predictions
    prediction_text = f"Food: {results['food_predictions'][0]['label']} ({results['food_predictions'][0]['probability']:.2f})\n"
    prediction_text += f"Weight: {results['predicted_weight']:.1f}g"
    
    plt.text(10, 30, prediction_text, color='white', fontsize=12, 
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
