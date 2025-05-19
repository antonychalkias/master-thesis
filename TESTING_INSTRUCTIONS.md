# Food Recognition and Weight Estimation Model - Testing Guide

This guide explains how to use the trained food recognition and weight estimation model (`best_model.pth`) for testing and inference.

## Model Overview

This model uses an EfficientNet-B0 architecture with dual output heads:
- A classification head for food recognition (11 classes)
- A regression head for weight estimation in grams

The model can identify the following food classes:
- apple
- avocado
- bagel
- biscuit
- blueberry_muffin
- broccoli
- chicken_nugget
- cinnamon_roll
- corn
- croissant
- strawberry

## Requirements

- Python 3.x
- PyTorch
- Matplotlib
- Pillow (PIL)
- torchvision
- numpy

## GPU Support

This model supports the following GPU acceleration:
- NVIDIA GPUs via CUDA
- Apple Silicon Macs via Metal Performance Shaders (MPS)

## Running Model Inference

### Option 1: Use the test_model.py script

This script provides a convenient way to test the model on individual images or directories of images.

#### Test a single image:

```bash
cd /Users/chalkiasantonios/Desktop/master-thesis/model-train-scripts
python test_model.py --model_path ../TRAIN_RESULTS/best_model.pth --image_path PATH_TO_IMAGE --output_dir ../results
```

#### Test multiple images in a directory:

```bash
cd /Users/chalkiasantonios/Desktop/master-thesis/model-train-scripts
python test_model.py --model_path ../TRAIN_RESULTS/best_model.pth --images_dir ../image_set_2 --output_dir ../results --limit 10
```

### Option 2: Use the original infer.py script

The original inference script provides more customization options.

```bash
cd /Users/chalkiasantonios/Desktop/master-thesis/model-train-scripts
python infer.py --model_path ../TRAIN_RESULTS/best_model.pth --image_path PATH_TO_IMAGE --output_dir ../results --use_gpu
```

### Option 3: Use the bbox_inference.py script for bounding boxes

This script adds bounding boxes around detected food items, which is useful for visualization and localization.

#### Add bounding boxes to a single image:

```bash
cd /Users/chalkiasantonios/Desktop/master-thesis/model-train-scripts
python bbox_inference.py --model_path ../TRAIN_RESULTS/best_model.pth --image_path PATH_TO_IMAGE --output_dir ../results_bbox
```

#### Add bounding boxes to multiple images:

```bash
cd /Users/chalkiasantonios/Desktop/master-thesis/model-train-scripts
python bbox_inference.py --model_path ../TRAIN_RESULTS/best_model.pth --images_dir ../image_set_2 --output_dir ../results_bbox --limit 5
```

## Command-line Arguments

- `--model_path`: Path to the trained model file (default: ../TRAIN_RESULTS/best_model.pth)
- `--image_path`: Path to a single food image for inference
- `--images_dir`: Directory containing multiple food images
- `--output_dir`: Directory to save results (default: ../results)
- `--num_classes`: Number of food classes (default: 11)
- `--limit`: Maximum number of images to process from a directory (default: 10)
- `--use_gpu`: Use GPU acceleration if available

## Output

The inference scripts generate two types of output for each image:
1. A visualization of the results (saved as a JPG file)
2. A JSON file with detailed prediction information

The visualization shows:
- The original image
- The predicted food class
- Confidence percentage
- Estimated weight in grams
- Top 3 predictions

### Bounding Box Output

When using the `bbox_inference.py` script, you'll get additional outputs:
1. An image with a bounding box around the detected food item
2. A combined visualization showing the original image and the detection side by side
3. A JSON file with the inference results

The bounding box visualization includes:
- A green rectangle around the detected food item
- A label showing the class name and confidence
- A weight estimate at the bottom of the box

## Example

```bash
# Process 5 images from the image_set_2 directory using GPU acceleration
cd /Users/chalkiasantonios/Desktop/master-thesis/model-train-scripts
python test_model.py --images_dir ../image_set_2 --output_dir ../results --limit 5

# Process 5 images with bounding boxes
cd /Users/chalkiasantonios/Desktop/master-thesis/model-train-scripts
python bbox_inference.py --images_dir ../image_set_2 --output_dir ../results_bbox --limit 5
```

## Troubleshooting

- If you encounter memory issues, reduce the batch size or use the `--limit` parameter
- If GPU acceleration is not working, try forcing CPU usage with the `--cpu` flag
- For Apple Silicon Macs, make sure you have a compatible version of PyTorch (2.0+)
