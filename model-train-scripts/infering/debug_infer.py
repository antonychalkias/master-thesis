#!/usr/bin/env python3
"""
Debug script for inference
"""

import os
import sys
import traceback

try:
    print("Python version:", sys.version)
    print("Working directory:", os.getcwd())
    print("Files in models:", os.listdir('./models'))
    print("Checking imports...")
    
    import torch
    print("PyTorch version:", torch.__version__)
    
    import numpy as np
    print("NumPy version:", np.__version__)
    
    # Fix for PyTorch 2.6+ serialization
    print("\nAdding numpy.core.multiarray.scalar to safe globals...")
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    
    print("\nChecking model implementation...")
    sys.path.append('./model-train-scripts')
    from model import MultiTaskNet
    print("Model class imported successfully")
    
    # Try to load the model file
    print("\nTrying to load model...")
    try:
        # Try with weights_only=False first
        checkpoint = torch.load("./models/best_model.pth", map_location=torch.device('cpu'), weights_only=False)
    except (RuntimeError, TypeError) as e:
        print(f"First attempt failed: {e}")
        # Try with safe globals
        checkpoint = torch.load("./models/best_model.pth", map_location=torch.device('cpu'))
    
    print("Model loaded successfully")
    print("Model keys:", checkpoint.keys())
    
    if 'model_state_dict' in checkpoint:
        print("model_state_dict exists with", len(checkpoint['model_state_dict']), "parameters")
        
    if 'label_to_idx' in checkpoint:
        print("label_to_idx exists with", len(checkpoint['label_to_idx']), "classes")
        print("Class labels:", list(checkpoint['label_to_idx'].keys())[:5], "...")
    
    print("\nCreating and loading model instance...")
    num_classes = len(checkpoint['label_to_idx'])
    model = MultiTaskNet(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded into instance successfully!")
    
    # Try running inference
    print("\nTrying to run inference on a sample image...")
    import torchvision.transforms as transforms
    from PIL import Image
    
    # Load a sample image
    sample_image_path = './images/1.jpeg'
    print(f"Loading image from {sample_image_path}")
    
    if os.path.exists(sample_image_path):
        image = Image.open(sample_image_path).convert('RGB')
        print(f"Image loaded, size: {image.size}")
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image).unsqueeze(0)
        print("Input tensor shape:", input_tensor.shape)
        
        with torch.no_grad():
            print("Running model inference...")
            class_logits, weight_pred = model(input_tensor)
            print("Inference completed!")
            
        print("Class logits shape:", class_logits.shape)
        print("Weight prediction:", weight_pred.item())
        
        # Convert to probabilities
        probabilities = torch.nn.functional.softmax(class_logits, dim=1)
        top_prob, top_class = torch.topk(probabilities, k=3, dim=1)
        
        # Convert indices to labels
        idx_to_label = {v: k for k, v in checkpoint['label_to_idx'].items()}
        top_classes = top_class.squeeze().cpu().numpy()
        
        print("\nTop 3 predictions:")
        for i, class_idx in enumerate(top_classes):
            label = idx_to_label[class_idx]
            prob = top_prob[0][i].item()
            print(f"{i+1}. {label}: {prob:.4f}")
    else:
        print(f"Sample image not found at {sample_image_path}")

except Exception as e:
    print("Error:", e)
    traceback.print_exc()
