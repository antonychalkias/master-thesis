# Food Recognition API

This API exposes the food recognition and weight estimation model through a simple HTTP interface.

## Features

- **Image Recognition**: Identify the type of food in an uploaded image
- **Weight Estimation**: Estimate the weight of the food item
- **Simple JSON Response**: Get predictions in an easy-to-parse JSON format

## Installation

### Prerequisites

- Python 3.6+
- PyTorch
- Torchvision
- Flask
- PIL (Pillow)
- NumPy
- Requests (for LLM fallback)

### Setup

1. Install the required dependencies:

```bash
pip install torch torchvision flask pillow numpy flask-cors requests
```

2. Make sure the model file is available at the specified path (default: `/Users/chalkiasantonios/Desktop/master-thesis/TRAIN_RESULTS/best_model.pth`)

3. If you want to use the LLM fallback feature, set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY=your_api_key_here
```

## Running the API Server

Navigate to the API directory and run the server:

```bash
cd /Users/chalkiasantonios/Desktop/master-thesis/api
python model_api.py --model_path /path/to/model.pth --port 5000
```

> **Note for macOS users**: Port 5000 is often used by AirPlay Receiver. If you encounter an "Address already in use" error, try using a different port (e.g., `--port 5001`) or disable AirPlay Receiver in System Settings → General → AirDrop & Handoff.

### Command Line Arguments

- `--model_path`: Path to the trained model file (default: `/Users/chalkiasantonios/Desktop/master-thesis/TRAIN_RESULTS/best_model.pth`)
- `--port`: Port to run the API server on (default: 5000)
- `--host`: Host to run the API server on (default: "0.0.0.0")
- `--num_classes`: Number of food classes in the model (default: 11)
- `--device`: Device to use for inference ("auto", "cpu", "cuda", "mps") (default: "auto")
- `--confidence_threshold`: Confidence threshold below which to use LLM fallback (default: 70.0)
- `--enable_llm_fallback`: Enable LLM fallback for low confidence predictions

## API Endpoints

### Health Check

`GET /health`

Check if the API is running and the model is loaded.

Example response:
```json
{
    "status": "ok",
    "message": "API is running"
}
```

### Predict

`POST /predict`

Upload an image for prediction.

**Request Format**: Form data with an image file field named 'image'.

**Query Parameters**:
- `confidence_threshold` (optional): Confidence threshold below which to use LLM fallback (default: 70.0)
- `use_llm_fallback` (optional): Whether to use LLM fallback for low confidence predictions (default: based on server configuration)

**Response Format**:
```json
{
    "class_idx": 0,
    "class_name": "apple",
    "confidence": 98.76,
    "weight": 150.5,
    "top_predictions": [
        {
            "label": "apple",
            "confidence": 98.76
        },
        {
            "label": "strawberry",
            "confidence": 0.87
        },
        {
            "label": "bagel",
            "confidence": 0.37
        }
    ],
    "verification": {
        "needed": false,
        "message": "Confidence is high enough, no verification needed."
    }
}
```

When confidence is low and LLM fallback is enabled, the response will include an additional `verification` field with the LLM's assessment.

## Testing the API

A test client is provided to demonstrate how to use the API:

```bash
cd /Users/chalkiasantonios/Desktop/master-thesis/api
python test_api.py --api_url http://localhost:5000 --image_path /path/to/image.jpg
```

### Command Line Arguments

- `--api_url`: URL of the API server (default: http://localhost:5000)
- `--image_path`: Path to the image file to send to the API (required)

## Error Handling

The API returns appropriate HTTP status codes for different types of errors:

- 400: Bad Request (e.g., no image provided, invalid image format)
- 500: Internal Server Error
- 503: Service Unavailable (e.g., model not loaded)

## Testing the LLM Fallback

The LLM fallback feature can be tested in several ways:

1. **Command line with test_api.py**:
```bash
python test_api.py --api_url http://localhost:5001 --image_path /path/to/image.jpg --confidence_threshold 85 --use_llm_fallback
```

2. **Web browser demo**:
   - Open `demo.html` in a web browser
   - Adjust the confidence threshold slider and enable LLM fallback
   - Upload an image and check the results

3. **Direct API calls**:
```bash
# With curl
curl -X POST -F "image=@/path/to/image.jpg" "http://localhost:5001/predict?confidence_threshold=80&use_llm_fallback=true"

# With Python requests
import requests

response = requests.post(
    "http://localhost:5001/predict",
    files={"image": open("/path/to/image.jpg", "rb")},
    params={"confidence_threshold": 80, "use_llm_fallback": "true"}
)
```

### Environment Variables

When using the LLM fallback feature, you need to set the `OPENAI_API_KEY` environment variable:

```bash
# Set temporarily for the current session
export OPENAI_API_KEY=your_api_key_here

# Or when starting the server
OPENAI_API_KEY=your_api_key_here python model_api.py
```

## Integration with Other Applications

You can integrate this API with web applications, mobile apps, or other services by making HTTP requests to the predict endpoint.

### Example with cURL

```bash
curl -X POST -F "image=@/path/to/image.jpg" http://localhost:5001/predict
```

### Example with Python Requests

```python
import requests

api_url = "http://localhost:5001/predict"
image_path = "/path/to/image.jpg"

with open(image_path, 'rb') as f:
    files = {'image': (image_path, f, 'image/jpeg')}
    response = requests.post(api_url, files=files)

if response.status_code == 200:
    prediction = response.json()
    print(f"Predicted food: {prediction['class_name']}")
    print(f"Estimated weight: {prediction['weight']}g")
```
