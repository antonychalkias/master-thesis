# Food Recognition and Weight Estimation Model for Kaggle

This package contains a multi-task learning model that performs food recognition and weight estimation from images. It uses a ResNet50 backbone with two specialized heads: one for food classification and another for weight regression.

## Overview

The model is designed to:
1. Identify food items from images (classification task)
2. Estimate their weight in grams (regression task)

## Files Included

- `kaggle_model.py`: The complete model code, combined from multiple modular files
- `kaggle_notebook.ipynb`: A sample notebook demonstrating how to use the model
- `requirements.txt`: List of dependencies needed to run the model

## Requirements

```
torch>=1.13.0
torchvision>=0.14.0
pandas>=1.5.0
numpy>=1.23.0
Pillow>=9.4.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

## Usage

### Quick Start

1. Upload the `kaggle_model.py` file to your Kaggle notebook
2. Import the necessary components:

```python
from kaggle_model import MultiTaskNet, prepare_data, get_device
```

3. Load or train a model:

```python
# Initialize model
model = MultiTaskNet(num_classes=47)
device = get_device()
model.to(device)
```

### Training

To train a new model on your dataset:

```python
from kaggle_model import prepare_data, MultiTaskNet, train_model, get_device

# Prepare data
train_dataloader, val_dataloader, label_to_idx = prepare_data(
    csv_path="path/to/your/data.csv", 
    images_dir="path/to/your/images",
    batch_size=16
)

# Initialize model
num_classes = len(label_to_idx)
device = get_device()
model = MultiTaskNet(num_classes)
model.to(device)

# Train
training_logs = train_model(model, train_dataloader, val_dataloader, device, num_epochs=20, model_save_dir="./")
```

### Inference

To use a trained model for inference:

```python
import torch
from kaggle_model import MultiTaskNet
from PIL import Image
import torchvision.transforms as transforms

# Load model
checkpoint = torch.load("best_model.pth", map_location="cpu")
label_to_idx = checkpoint["label_to_idx"]
idx_to_label = {v: k for k, v in label_to_idx.items()}

model = MultiTaskNet(num_classes=len(label_to_idx))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Process an image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open("food_image.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# Get predictions
with torch.no_grad():
    class_logits, weight_pred = model(input_tensor)

# Process results
probabilities = torch.nn.functional.softmax(class_logits, dim=1)
top_prob, top_class = torch.topk(probabilities, k=3, dim=1)
top_class_idx = top_class.squeeze().cpu().numpy()[0]
predicted_food = idx_to_label[top_class_idx]
predicted_weight = float(weight_pred.item())

print(f"Food: {predicted_food}, Weight: {predicted_weight:.1f}g")
```

## Dataset Format

The model expects a CSV file with the following columns:
- `image_name`: Name of the image file
- `labels`: Food category label
- `weight`: Weight in grams

## Model Architecture

- Backbone: ResNet50 pretrained on ImageNet
- Classification Head: Linear layer with softmax activation
- Regression Head: Linear layer for weight estimation

## License

This model is provided for educational and research purposes.

---

For more information and the complete project, please visit the original GitHub repository.
