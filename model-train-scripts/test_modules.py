#!/usr/bin/env python3
"""
Test script to verify the modularized code works correctly.
This script will:
1. Import the necessary modules
2. Print module versions and availability
3. Create a small model and verify forward pass
"""

import os
import sys
import torch
import importlib.util

def check_module_exists(module_name, file_path):
    """Check if a Python module exists and can be imported"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"✅ {module_name} module loaded successfully from {file_path}")
        return True
    except Exception as e:
        print(f"❌ Error loading {module_name} module: {e}")
        return False

def main():
    # Current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Print Python and PyTorch versions
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print("\n")
    
    # Check if all modules exist and can be loaded
    modules = [
        ("data", os.path.join(current_dir, "data.py")),
        ("model", os.path.join(current_dir, "model.py")),
        ("utils", os.path.join(current_dir, "utils.py")),
        ("train", os.path.join(current_dir, "train.py"))
    ]
    
    all_modules_loaded = True
    for module_name, file_path in modules:
        if not check_module_exists(module_name, file_path):
            all_modules_loaded = False
    
    if not all_modules_loaded:
        print("Some modules failed to load. Please check the errors above.")
        return
    
    # Create a small model to test forward pass
    try:
        from model import MultiTaskNet
        
        print("\nTesting model forward pass...")
        model = MultiTaskNet(num_classes=10)
        dummy_input = torch.randn(2, 3, 224, 224)  # [batch_size, channels, height, width]
        class_logits, weight_pred = model(dummy_input)
        
        print(f"Model forward pass successful:")
        print(f" - Classification output shape: {class_logits.shape}")
        print(f" - Weight prediction shape: {weight_pred.shape}")
        print("\nAll checks passed! The modularized code structure appears to be working correctly.")
    
    except Exception as e:
        print(f"Error testing model: {e}")

if __name__ == "__main__":
    main()
