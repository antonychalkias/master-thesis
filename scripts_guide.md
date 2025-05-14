# Model Scripts Usage Guide

This document provides a quick reference guide for all the scripts in the model-train-scripts directory, explaining what each file does and how to run it.

## Training & Inference Scripts

### `train.py`
**Purpose**: Main script for training the food recognition and weight estimation model.  
**Usage**:
```bash
# Basic usage with default parameters (One Cycle LR)
python train.py --csv_path ../csvfiles/latest.csv --images_dir ../images --model_dir ../models/my_model

# With specific learning rate strategy
python train.py --lr_strategy one_cycle --epochs 30 --batch_size 16 --lr 5e-5
```

### `infer.py`
**Purpose**: Run inference on new images using a trained model.  
**Usage**:
```bash
# Infer on a single image
python infer.py --model_path ../../models/my_model/best_model.pth --image_path ../../images/120.jpg

# Infer on a directory of images
python infer.py --model_path ../../models/my_model/best_model.pth --images_dir ../../images/ --output_dir ../../results/
```

## Learning Rate Tuning Scripts

### `lr_finder.py`
**Purpose**: Automatically find the optimal learning rate for your dataset.  
**Usage**:
```bash
# Use directly
python lr_finder.py --csv_path ../csvfiles/latest.csv --images_dir ../images --model_dir ../models/lr_test

# Or through train.py
python train.py --lr_strategy find
```

### `one_cycle_lr.py`
**Purpose**: Contains the One Cycle Learning Rate scheduler implementation.  
**Note**: This is imported by other scripts and not meant to be run directly.

## Visualization & Analysis Scripts

### `visualize_training_advanced.py`
**Purpose**: Create detailed visualization plots for training metrics, including learning rate analysis.  
**Usage**:
```bash
python visualize_training_advanced.py --log_path ../models/my_model/training_log.json --output_dir ../models/my_model/plots
```

### `compare_lr_strategies.py`
**Purpose**: Train models with different learning rate strategies and compare their performance.  
**Usage**:
```bash
# Compare multiple strategies
python compare_lr_strategies.py --csv_path ../csvfiles/latest.csv --images_dir ../images --strategies one_cycle cosine step

# Customize epochs and batch size
python compare_lr_strategies.py --epochs 10 --batch_size 8 --lr 5e-5 --strategies one_cycle step
```

## Core Model Files

### `model.py`
**Purpose**: Defines the model architecture (MultiTaskNet with ResNet50 backbone).  
**Note**: This is imported by other scripts and not meant to be run directly.

### `data.py`
**Purpose**: Contains dataset implementation and data loading functionality.  
**Note**: This is imported by other scripts and not meant to be run directly.

### `utils.py`
**Purpose**: Contains utility functions for argument parsing and device selection.  
**Note**: This is imported by other scripts and not meant to be run directly.

## Quick Reference

| Task | Script to Run |
|------|--------------|
| Train a model | `python train.py` |
| Run inference | `cd model-train-scripts/infering && python infer.py` |
| Find optimal learning rate | `python train.py --lr_strategy find` |
| Visualize training results | `python visualize_training_advanced.py` |
| Compare learning rate strategies | `python compare_lr_strategies.py` |

## Example Workflow

Note: The following commands will create the necessary directories if they don't exist yet.

```bash
# 1. Find the optimal learning rate
cd model-train-scripts
python train.py --lr_strategy find --model_dir ../models/lr_finder_test

# 2. Train with the suggested learning rate using One Cycle LR
python train.py --lr 5e-5 --lr_strategy one_cycle --model_dir ../models/my_training_run

# 3. Train with One Cycle LR and save to TRAIN_RESULTS
python train.py --csv_path ../csvfiles/latest.csv --images_dir ../images --model_dir ../TRAIN_RESULTS --epochs 20 --batch_size 16 --lr 5e-5 --lr_strategy one_cycle

# 4. Visualize the training results
cd create_plots
python visualize_training_advanced.py --log_path ../../TRAIN_RESULTS/training_log_2.json --output_dir ../../TRAIN_RESULTS/plots

# 5. Run inference on new images
cd ../infering
# If model.py symlink doesn't exist already:
# ln -s ../model.py model.py  # Create symlink to model.py (only needed once)
python infer.py --model_path ../../TRAIN_RESULTS/best_model.pth --images_dir ../../images --output_dir ../../results

# 6. Run inference on subset of images (10 random images)
mkdir -p ../../sample_10_images
find ../../images -name "*.jpg" | sort -R | head -10 | xargs -I {} cp {} ../../sample_10_images/
python infer.py --model_path ../../TRAIN_RESULTS/best_model.pth --images_dir ../../sample_10_images --output_dir ../../results_10_images

# 7. Visualize specific training results (if you want to visualize a different log file)
cd ../create_plots
python visualize_training_advanced.py --log_path ../../TRAIN_RESULTS/training_log_2.json --output_dir ../../TRAIN_RESULTS/plots
```
