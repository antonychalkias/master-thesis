# 🍽️ Food Image Recognition, Weight Estimation & Nutrition Tool

This project provides a complete workflow for labeling, segmenting, and preparing datasets for machine learning models that perform **food image recognition**, **weight estimation**, and **nutritional analysis**.

---

## 📚 Table of Contents

### 📋 Part 1: Data Labeling & Processing
- [🔧 Setup](#-setup)
- [🚀 Steps to Use](#-steps-to-use)
  - [1. Install Label Studio](#1-install-label-studio)
  - [2. Start Label Studio](#2-start-label-studio)
  - [3. Generate Labeling JSON](#3-generate-labeling-json)
  - [4. Prepare Dataset for Labeling](#4-prepare-dataset-for-labeling)
  - [5. Serve Images](#5-serve-images)
  - [6. Import JSON to Label Studio](#6-import-json-to-label-studio)
  - [7. Label Studio Template Format](#7-label-studio-template-format)
- [📝 Semantic Segmentation Labels](#-semantic-segmentation-labels)
- [🔄 Dataset Processing](#-dataset-processing)
  - [1. Combining CSV Datasets](#1-combining-csv-datasets)
  - [2. Reordering and Renaming Images](#2-reordering-and-renaming-images)

### 🧠 Part 2: Model Training & Inference
- [📊 Model Architecture](#-model-architecture)
- [⚙️ Training Process](#-training-process)
- [🔍 Evaluation](#-evaluation)
- [🖼️ Inference](#-inference)

---

## 🔧 Setup

To get started, make sure you have Python installed and then install [Label Studio](https://labelstud.io/):

```bash
pip install label-studio
```

---

## 🚀 Steps to Use

### 1. Install Label Studio

```bash
pip install label-studio
```

### 2. Start Label Studio

```bash
label-studio start
```

---

### 3. Generate Labeling JSON

To overcome some Label Studio limitations, we've provided a custom script to generate the labeling JSON:

```bash
cd python_scripts
python3 generate_images_to_json.py
```

---

### 4. Prepare Dataset for Labeling

After generating the JSON, run Label Studio again:

```bash
label-studio start
```

---

### 5. Serve Images

Run the `server.py` script to serve your images with CORS enabled:

```bash
python3 server.py
```

---

### 6. Import JSON to Label Studio

In the Label Studio UI:
- Create or open a project.
- Go to **Import** and load the generated `.json` file from the previous steps.

---

### 7. Label Studio Template Format

For **semantic segmentation** with polygon labels, use the following configuration:

```xml
<View>
  <Image name="image" value="$image" zoom="true"/>

  <PolygonLabels name="label" toName="image" strokeWidth="3">
    <Label value="Gemista" background="red"/>
    <Label value="Green Beans" background="green"/>
    <Label value="Burgers" background="brown"/>
    <Label value="Chicken" background="orange"/>
    <Label value="Giouvetsi" background="purple"/>
    <Label value="Feta" background="lightblue"/>
    <Label value="Cucumber" background="darkgreen"/>
    <Label value="Pasta" background="yellow"/>
    <Label value="Minced Meat" background="maroon"/>
    <Label value="Cheese" background="gold"/>
    <Label value="Rice" background="pink"/>
    <Label value="Okra" background="olive"/>
    <Label value="Toast Cheese" background="beige"/>
    <Label value="Eggs" background="lightyellow"/>
    <Label value="Lentils" background="saddlebrown"/>
    <Label value="Salad Leaves" background="lightgreen"/>
    <Label value="Mushrooms" background="#A0522D"/>
    <Label value="Other" background="gray"/>
    <Label value="Bread" background="#FFA39E"/>
    <Label value="Potatos" background="#FFA46E"/>
    <Label value="Carrots" background="orange"/> 
    <Label value="Onions" background="#B7EB8F"/>
    <Label value="Peas" background="#B7E19F"/>
    <Label value="Snails" background="#A0500D"/>
    <Label value="Lamb" background="#AA00DD"/>
    <Label value="Broccoli" background="#90EE90"/>
    <Label value="Corn" background="yellow"/>
    
  </PolygonLabels>

  <TextArea name="other_description" toName="image"
            perRegion="true"
            visibleWhen="choice=Other"
            required="false"
            placeholder="Please describe the 'Other' item"
            rows="3"/>

  <TextArea name="totalWeight" toName="image"
            editable="true"
            required="true"
            maxSubmissions="1"
            placeholder="Enter total plate weight (e.g., 100g)"
            rows="1"/>
</View>
```

---

## 📝 Semantic Segmentation Labels

These are the predefined food categories for polygon annotation:

  - Gemista 🫑
  - Green Beans 🫘
  - Burgers 🍔
  - Chicken 🍗
  - Giouvetsi 🍲
  - Feta 🧀
  - Cucumber 🥒
  - Pasta 🍝
  - Minced Meat 🥩
  - Cheese 🧀
  - Rice 🍚
  - Okra 🌿
  - Toast Cheese 🥪
  - Eggs 🥚
  - Lentils 🍛
  - Salad Leaves 🥗
  - Mushrooms 🍄
  - Other ❓
  - Bread 🍞
  - Potatos 🥔
  - Carrots 🥕
  - Onions 🧅
  - Peas 🌱
  - Snails 🐌
  - Lamb 🐑
  - Broccoli 🥦
  - Corn 🌽


---

## 🔄 Dataset Processing

After labeling your food images, use these scripts to process and prepare your datasets for training machine learning models.

### 1. Combining CSV Datasets

The `combine_csv_datasets.py` script merges data from multiple CSV files into a single comprehensive dataset:

```bash
cd python_scripts/after_labeling_scripts
python3 combine_csv_datasets.py
```

This script:
- Combines data from `labels_for_ordered_dataset.csv` and `labels_for_dataset_foods.csv`
- Extracts food labels from polygon annotations in JSON format
- Processes image names and URLs to maintain consistency
- Creates a unified dataset with fields for image name, labels, weight, volume, energy, etc.
- Outputs a combined CSV file at `csvfiles/combined_dataset_labels.csv`

### 2. Reordering and Renaming Images

The `reorder_images_update_csv.py` script renames all images sequentially and updates the CSV references:

```bash
cd python_scripts
python3 reorder_images_update_csv.py
```

This script:
- Combines images from `dataset_foods` and `ordered_dataset` folders
- Renames all images sequentially (1.jpg, 2.jpg, etc.)
- Creates a new directory `ordered_dataset_foods_ready` with the renamed images
- Updates the CSV file to reference the new image names
- Preserves original image names in the CSV for reference
- Creates a new CSV file at `csvfiles/combined_dataset_labels_ready.csv`

---

## 📸 Image Data Access

For the dataset of annotated images contact me.

---

## 📊 Model Architecture

The model uses a multi-task learning approach built on a ResNet50 backbone:
- **Backbone**: ResNet50 pretrained on ImageNet
- **Heads**: 
  - Classification head for food recognition (47 food categories)
  - Regression head for weight estimation

### Code Structure

The model code is organized into the following modules:
- `train.py` - Main training script with the training loop
- `model.py` - Model architecture definition (MultiTaskNet)
- `data.py` - Dataset implementation and data loading functionality
- `utils.py` - Utility functions for argument parsing and device selection
- `lr_finder.py` - Learning rate finder implementation
- `one_cycle_lr.py` - Custom One Cycle Learning Rate scheduler
- `visualize_training_advanced.py` - Advanced training metrics visualization
- `compare_lr_strategies.py` - Tool to compare different learning rate strategies

---

## 🏃 Running the Model

### Training the Model

To train the model using your dataset of food images:

```bash
# Navigate to the model-train-scripts directory
cd /Users/chalkiasantonios/Desktop/master-thesis/model-train-scripts

# Run the training script with default parameters (One Cycle LR strategy)
python train.py

# Run with custom parameters and specific learning rate strategy
python train.py --epochs 30 --batch_size 16 --lr 5e-5 --lr_strategy one_cycle --num_workers 4

# Try the learning rate finder to determine the optimal learning rate
python train.py --lr_strategy find

# Use cosine annealing learning rate scheduler
python train.py --lr_strategy cosine

# Use step learning rate scheduler
python train.py --lr_strategy step
```

#### Available Parameters:
- `--csv_path`: Path to the CSV file with annotations (default: "latest_lab.csv")
- `--images_dir`: Path to the directory with images (default: "../images")
- `--model_dir`: Directory to save the model (default: "../models")
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size for training (default: 16)
- `--lr`: Learning rate (default: 5e-5)
- `--lr_strategy`: Learning rate strategy (choices: "one_cycle", "cosine", "step", "find", default: "one_cycle")
- `--num_workers`: Number of workers for data loading (default: 0)

### Model Inference

To use the trained model for inference on new images:

```bash
# Navigate to the model-train-scripts directory
cd /Users/chalkiasantonios/Desktop/master-thesis/model-train-scripts

# Run inference on a single image
python infer.py --model_path ../models/best_model.pth --image_path ../path/to/food_image.jpg

# Process a directory of images
python infer.py --model_path ../models/best_model.pth --images_dir ../path/to/images/ --output_dir ../results/
```

#### Inference Parameters:
- `--model_path`: Path to the trained model (required)
- `--image_path`: Path to a single food image
- `--images_dir`: Directory with multiple food images
- `--output_dir`: Directory to save results (default: "results")

### Troubleshooting Common Issues

1. **Missing packages**: Ensure all required packages are installed:
   ```bash
   pip install torch torchvision pandas pillow scikit-learn matplotlib
   ```

2. **CUDA/MPS issues**: The code automatically detects and uses available acceleration (CUDA for NVIDIA GPUs, MPS for Apple Silicon).

3. **Image loading errors**: The model handles case-sensitive file extensions and will create placeholder images for missing files.

4. **Memory issues**: Reduce batch size if you encounter memory problems.

---

## ⚙️ Training Process

### Environment Setup

```bash
# Install pandas
  pip install torch torchvision pandas pillow
```

### Data Preparation

```bash
# Prepare the dataset for training
python prepare_dataset.py --input_csv csvfiles/combined_dataset_labels_ready.csv --images_dir ordered_dataset_foods_ready
```

### Training

```bash
# Start the training process
python train.py --config configs/default_config.yaml --epochs 100 --batch_size 8
```

### Training Parameters

- **Optimizer**: Adam with learning rate 5e-5 (default)
- **Loss Functions**: 
  - Segmentation: Focal Loss + Dice Loss
  - Classification: Cross-Entropy Loss
  - Weight Estimation: Mean Absolute Error
- **Batch Size**: 16 (default)
- **Epochs**: 20 (default)
- **Validation Split**: 20%
- **Learning Rate Strategies**:
  - One Cycle LR (default): Dynamic learning rate that first increases then decreases
  - Cosine Annealing: Gradual decay using cosine function
  - Step LR: Gradual step-based decay (step_size=5, gamma=0.75)
  - Learning Rate Finder: Automatically determine optimal learning rate

---

## 🔍 Evaluation

The model is evaluated on several metrics:

```bash
# Run evaluation on test set
python evaluate.py --model_path models/best_model.pth --test_csv csvfiles/test_set.csv
```

### Performance Metrics

- **Segmentation**: mIoU (mean Intersection over Union) and Dice coefficient
- **Classification**: Accuracy, Precision, Recall, and F1-score
- **Weight Estimation**: MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error)

---

## 🖼️ Inference

The trained model can be used for inference on new images:

```bash
# Run inference on a single image
python infer.py --model_path models/best_model.pth --image_path path/to/food_image.jpg

# Process a directory of images
python infer.py --model_path models/best_model.pth --images_dir path/to/images/ --output_dir path/to/results/
```

### Inference Output

The model outputs:
- Segmentation mask for each food item
- Food type classification for each segment
- Estimated weight for each food item
- Total caloric content estimation

Sample visualization is saved along with detailed nutritional information in JSON format.

