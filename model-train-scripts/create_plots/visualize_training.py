#!/usr/bin/env python3
"""
Script to visualize training progress from training logs.
"""

import json
import matplotlib.pyplot as plt
import os

# Load training log
log_path = '../models/training_log.json'
with open(log_path, 'r') as f:
    logs = json.load(f)

# Create output directory for plots
os.makedirs('plots', exist_ok=True)

# Plot training and validation loss
plt.figure(figsize=(12, 8))

# Plot 1: Loss Curves
plt.subplot(2, 2, 1)
plt.plot(logs['epochs'], logs['train_loss'], 'b-', label='Training Loss')
plt.plot(logs['epochs'], logs['val_loss'], 'r-', label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Plot 2: Validation Accuracy
plt.subplot(2, 2, 2)
plt.plot(logs['epochs'], logs['val_accuracy'], 'g-')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)

# Plot 3: Weight Mean Absolute Error
plt.subplot(2, 2, 3)
plt.plot(logs['epochs'], logs['weight_mae'], 'm-')
plt.title('Weight Estimation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE (grams)')
plt.grid(True)

# Plot 4: Loss values (log scale to see details better)
plt.subplot(2, 2, 4)
plt.semilogy(logs['epochs'], logs['train_loss'], 'b-', label='Training Loss')
plt.semilogy(logs['epochs'], logs['val_loss'], 'r-', label='Validation Loss')
plt.title('Loss Curves (Log Scale)')
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('plots/training_progress.png', dpi=300)

# Create a second figure for training metrics comparison
plt.figure(figsize=(10, 6))
plt.plot(logs['epochs'], logs['val_accuracy'], 'g-', label='Validation Accuracy')
plt.plot(logs['epochs'], [mae/max(logs['weight_mae']) for mae in logs['weight_mae']], 'm-', 
         label='Normalized Weight MAE')
plt.title('Training Metrics Comparison')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.grid(True)
plt.legend()
plt.savefig('plots/metrics_comparison.png', dpi=300)

print(f"Plots saved to plots/training_progress.png and plots/metrics_comparison.png")
