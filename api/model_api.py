#!/usr/bin/env python3
"""
Flask API for Food Recognition and Weight Estimation Model.
This API allows image upload and returns model predictions including food class and weight.
"""

import os
import sys
import torch
import json
import requests
from PIL import Image
import numpy as np
import io
import base64
from torchvision import transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for matplotlib

# Add the model-train-scripts directory to the path to import the model
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(os.path.dirname(script_dir), 'model-train-scripts'))

try:
    from model import MultiTaskNet
except ImportError:
    print("Error: Cannot import MultiTaskNet. Make sure the model-train-scripts directory is in the path.")
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for model and device
model = None
idx_to_label = None
device = 'cpu'
config = {
    'confidence_threshold': 70.0,
    'enable_llm_fallback': False
}

def load_model(model_path, num_classes=11, target_device='cpu'):
    """Load a trained model for inference."""
    global model, idx_to_label, device
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # Set device based on availability
        if target_device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'  # Use MPS on macOS with Apple Silicon
            else:
                device = 'cpu'
        else:
            device = target_device
            
        print(f"Using device: {device}")
        
        # Load checkpoint with weights_only=True for security
        try:
            checkpoint = torch.load(model_path, map_location=torch.device(device), weights_only=True)
        except TypeError:
            # Fall back for older PyTorch versions that don't support weights_only
            checkpoint = torch.load(model_path, map_location=torch.device(device))
        
        # Load label mapping if available in the checkpoint
        label_to_idx = checkpoint.get('label_to_idx', None)
        
        # Create model instance
        model = MultiTaskNet(num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully from {model_path}")
        
        if label_to_idx:
            print(f"Found label mapping with {len(label_to_idx)} classes")
            # Create reverse mapping for inference
            idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        else:
            print("No label mapping found in model checkpoint. Using hardcoded labels.")
            # Hardcoded labels from training_results.json
            food_classes = [
                "apple", "avocado", "bagel", "biscuit", "blueberry_muffin", 
                "broccoli", "chicken_nugget", "cinnamon_roll", "corn", 
                "croissant", "strawberry"
            ]
            idx_to_label = {i: label for i, label in enumerate(food_classes)}
        
        return True
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def prepare_image(image_data):
    """
    Prepare an image for inference by applying appropriate transformations.
    
    Args:
        image_data: Image data in bytes or PIL Image
        
    Returns:
        torch.Tensor: Processed image tensor ready for model inference
        PIL.Image: Original image
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
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        else:
            image = image_data.convert('RGB')
            
        transformed_image = transform(image)
        
        # Add batch dimension
        return transformed_image.unsqueeze(0), image
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

def run_inference(image_tensor):
    """
    Run inference on an image tensor using the loaded model.
    
    Args:
        image_tensor: Preprocessed image tensor
        
    Returns:
        dict: Inference results including class prediction and weight estimation
    """
    global model, idx_to_label, device
    
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
                {"label": idx_to_label.get(idx.item(), f"Unknown_{idx.item()}"), 
                 "confidence": float(prob.item() * 100)} 
                for idx, prob in zip(top_indices, top_values)
            ]
            
            return {
                'class_idx': int(predicted_class_idx),
                'class_name': predicted_label,
                'confidence': float(confidence * 100),  # Convert to percentage
                'weight': float(predicted_weight),
                'top_predictions': top_predictions
            }
    
    except Exception as e:
        print(f"Error during inference: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify API is running"""
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 503
    return jsonify({'status': 'ok', 'message': 'API is running'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to receive an image, process it with the model, and return predictions.
    Accepts either a file upload or a base64 encoded image.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded. Please initialize the model first.'}), 503
    
    try:
        # Check if the post request has the file part
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided. Please upload an image file.'}), 400
        
        image_file = request.files['image']
        
        # If the user submits an empty form
        if image_file.filename == '':
            return jsonify({'error': 'No image selected. Please select an image file.'}), 400
        
        # Process the image and run inference
        image_data = image_file.read()
        image_tensor, original_image = prepare_image(image_data)
        
        if image_tensor is None:
            return jsonify({'error': 'Error processing image. Please try another image.'}), 400
        
        # Run inference
        result = run_inference(image_tensor)
        
        if result is None:
            return jsonify({'error': 'Error during inference. Please try again.'}), 500
        
        # If confidence is low, use LLM fallback
        # Override global config with query parameters if provided
        confidence_threshold = float(request.args.get('confidence_threshold', config['confidence_threshold']))
        use_llm_fallback = request.args.get('use_llm_fallback', str(config['enable_llm_fallback'])).lower() == 'true'
        
        if use_llm_fallback and result['confidence'] < confidence_threshold:
            # Use the original image data for LLM
            result = query_llm_for_verification(image_data, result, confidence_threshold)
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

def initialize_model(model_path, num_classes=11, target_device='auto'):
    """Initialize the model when starting the API server"""
    return load_model(model_path, num_classes, target_device)

def query_llm_for_verification(image_data, prediction_result, confidence_threshold=70.0):
    """
    Query an LLM as a fallback when confidence is below threshold.
    
    Args:
        image_data: The original image data (bytes)
        prediction_result: The initial prediction result from the model
        confidence_threshold: Threshold below which to use LLM fallback
        
    Returns:
        dict: Enhanced prediction result with LLM verification
    """
    # Only query LLM if confidence is below threshold
    if prediction_result['confidence'] >= confidence_threshold:
        # Add verification field but mark as not needed
        prediction_result['verification'] = {
            'needed': False,
            'message': "Confidence is high enough, no verification needed."
        }
        return prediction_result
    
    try:
        # For this example, we're using OpenAI's API
        # Replace with your preferred LLM API
        OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
        
        if not OPENAI_API_KEY:
            prediction_result['verification'] = {
                'needed': True,
                'success': False,
                'message': "LLM verification needed but API key not configured"
            }
            return prediction_result
            
        # Convert image to base64 for API submission
        if isinstance(image_data, bytes):
            base64_image = base64.b64encode(image_data).decode('utf-8')
        else:
            # If it's a PIL image, convert to bytes first
            buffer = io.BytesIO()
            image_data.save(buffer, format="JPEG")
            base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
        # Prepare request for OpenAI API
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {OPENAI_API_KEY}'
        }
        
        # Create prompt for the LLM
        prompt = f"""
        I have an image of food that my model has classified as '{prediction_result['class_name']}' 
        with {prediction_result['confidence']:.2f}% confidence.
        
        The estimated weight is {prediction_result['weight']:.2f}g.
        
        The top alternative predictions are:
        {', '.join([f"{pred['label']} ({pred['confidence']:.2f}%)" for pred in prediction_result['top_predictions']])}
        
        Can you verify if this classification is correct based on the image?
        If it's not correct, what do you think it is instead?
        """
        
        # Build the API request
        payload = {
            'model': 'gpt-4-vision-preview',  # Use vision-capable model
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            'max_tokens': 300
        }
        
        # Make the API request
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            llm_response = response.json()
            llm_text = llm_response['choices'][0]['message']['content']
            
            # Add LLM verification to the result
            prediction_result['verification'] = {
                'needed': True,
                'success': True,
                'llm_response': llm_text
            }
        else:
            # Handle API error
            prediction_result['verification'] = {
                'needed': True,
                'success': False,
                'message': f"LLM API error: {response.status_code} - {response.text}"
            }
            
    except Exception as e:
        # Handle any exceptions
        prediction_result['verification'] = {
            'needed': True,
            'success': False,
            'message': f"Error during LLM verification: {str(e)}"
        }
    
    return prediction_result

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the Food Recognition API server")
    parser.add_argument("--model_path", type=str, 
                        default="/Users/chalkiasantonios/Desktop/master-thesis/TRAIN_RESULTS/best_model.pth",
                        help="Path to the trained model")
    parser.add_argument("--port", type=int, default=5001, 
                        help="Port to run the API server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                        help="Host to run the API server on")
    parser.add_argument("--num_classes", type=int, default=11, 
                        help="Number of food classes")
    parser.add_argument("--device", type=str, default="auto", 
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Device to use for inference")
    parser.add_argument("--confidence_threshold", type=float, default=70.0,
                        help="Confidence threshold below which to use LLM fallback")
    parser.add_argument("--enable_llm_fallback", action="store_true",
                        help="Enable LLM fallback for low confidence predictions")
    
    args = parser.parse_args()
    
    # Store config values
    config['confidence_threshold'] = args.confidence_threshold
    config['enable_llm_fallback'] = args.enable_llm_fallback
    
    # Initialize the model
    if initialize_model(args.model_path, args.num_classes, args.device):
        print("Model initialized successfully. Starting API server...")
        
        # Run the Flask app
        app.run(host=args.host, port=args.port, debug=False)
    else:
        print("Failed to initialize model. Exiting.")
        sys.exit(1)
