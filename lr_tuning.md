# Food Recognition and Weight Estimation Model

This directory contains the machine learning model for food recognition and weight estimation.

## Model Overview

The model uses a multi-task learning approach with a ResNet50 backbone:
- Classification task: Food type recognition
- Regression task: Food weight estimation

## Files Description

- `train.py`: Main training script with the training loop
- `model.py`: Model architecture definition (MultiTaskNet)
- `data.py`: Dataset implementation and data loading functionality
- `utils.py`: Utility functions for argument parsing and device selection
- `lr_finder.py`: Learning rate finder implementation
- `one_cycle_lr.py`: Custom One Cycle Learning Rate scheduler
- `visualize_training_advanced.py`: Advanced training metrics visualization
- `compare_lr_strategies.py`: Tool to compare different learning rate strategies

## Learning Rate Tuning

This model supports several learning rate tuning strategies:

1. **One Cycle Learning Rate (Default)**: 
   - Increases learning rate then decreases it
   - Improves convergence speed and final performance
   - Command: `python train.py --lr_strategy one_cycle`

2. **Cosine Annealing**:
   - Gradually reduces learning rate using a cosine function
   - Command: `python train.py --lr_strategy cosine`

3. **Step Decay**:
   - Reduces learning rate by a factor at regular intervals
   - Command: `python train.py --lr_strategy step`

4. **Learning Rate Finder**:
   - Automatically determines the optimal learning rate
   - Command: `python train.py --lr_strategy find`

## Visualizing Results

To visualize training results:
```bash
python visualize_training_advanced.py --log_path ../models/your_model/training_log.json --output_dir ../models/your_model/plots
```

## Comparing Learning Rate Strategies

To compare different learning rate strategies:
```bash
python compare_lr_strategies.py --csv_path ../csvfiles/latest.csv --images_dir ../images --strategies one_cycle cosine step
```

This script will:
1. Train models with each specified strategy
2. Create plots comparing performance
3. Generate a summary of best metrics for each strategy
