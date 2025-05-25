#!/usr/bin/env python3
"""
Client script to test the Food Recognition API.
This script demonstrates how to send images to the API and receive predictions.
"""

import requests
import argparse
import os
import json
from pprint import pprint

def test_api_health(api_url):
    """Test the API health endpoint"""
    try:
        response = requests.get(f"{api_url}/health")
        if response.status_code == 200:
            print("API is running and healthy!")
            return True
        else:
            print(f"API health check failed with status code: {response.status_code}")
            print(response.json())
            return False
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return False

def predict_image(api_url, image_path, confidence_threshold=None, use_llm_fallback=None):
    """
    Send an image to the API for prediction
    
    Args:
        api_url: Base URL of the API
        image_path: Path to the image file
        confidence_threshold: Optional threshold for LLM fallback
        use_llm_fallback: Whether to enable LLM fallback
    
    Returns:
        API response as JSON or None if an error occurred
    """
    try:
        # Check if the file exists
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found")
            return None
        
        # Build URL with query parameters
        url = f"{api_url}/predict"
        params = {}
        if confidence_threshold is not None:
            params['confidence_threshold'] = confidence_threshold
        if use_llm_fallback is not None:
            params['use_llm_fallback'] = str(use_llm_fallback).lower()
        
        # Prepare the file for upload
        with open(image_path, 'rb') as f:
            files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
            
            # Send the request to the API
            response = requests.post(url, files=files, params=params)
            
            # Check if the request was successful
            if response.status_code == 200:
                return response.json()
            else:
                print(f"API request failed with status code: {response.status_code}")
                print(response.json())
                return None
    
    except Exception as e:
        print(f"Error sending request to API: {e}")
        return None

def main():
    """Main function to parse arguments and call the API"""
    parser = argparse.ArgumentParser(description="Test the Food Recognition API")
    parser.add_argument("--api_url", type=str, default="http://localhost:5000",
                        help="URL of the API server")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the image file to send to the API")
    parser.add_argument("--confidence_threshold", type=float,
                        help="Confidence threshold for LLM fallback")
    parser.add_argument("--use_llm_fallback", action="store_true",
                        help="Enable LLM fallback for low confidence predictions")
    
    args = parser.parse_args()
    
    # Check API health
    if not test_api_health(args.api_url):
        print("Exiting due to API health check failure")
        return
    
    # Send the image for prediction
    result = predict_image(
        args.api_url, 
        args.image_path,
        confidence_threshold=args.confidence_threshold,
        use_llm_fallback=args.use_llm_fallback if args.use_llm_fallback else None
    )
    
    # Display the results
    if result:
        print("\nPrediction Results:")
        print(f"Predicted Food: {result['class_name']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"Estimated Weight: {result['weight']:.2f}g")
        
        print("\nTop Predictions:")
        for i, pred in enumerate(result['top_predictions']):
            print(f"{i+1}. {pred['label']}: {pred['confidence']:.2f}%")
        
        # Display verification info if available
        if 'verification' in result:
            print("\nVerification:")
            verification = result['verification']
            if verification['needed']:
                print("Low confidence detected - LLM verification was needed")
                if verification['success']:
                    print("\nLLM Response:")
                    print(verification['llm_response'])
                else:
                    print(f"LLM verification failed: {verification.get('message', 'Unknown error')}")
            else:
                print(verification['message'])
        
        # Save result to a JSON file
        output_path = f"prediction_{os.path.basename(args.image_path)}.json"
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"\nPrediction saved to {output_path}")

if __name__ == "__main__":
    main()
