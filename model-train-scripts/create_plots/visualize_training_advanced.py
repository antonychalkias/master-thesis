#!/usr/bin/env python3
"""
Script to visualize training results with learning rate analysis.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_training_metrics(log_path, output_dir):
    """
    Create plots for training metrics including learning rate analysis.
    
    Args:
        log_path: Path to the training log JSON file
        output_dir: Directory to save the output plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training logs
    with open(log_path, 'r') as f:
        logs = json.load(f)
    
    # Verify that the logs contain the expected data
    required_keys = ['epochs', 'train_loss', 'val_loss', 'val_accuracy', 'weight_mae']
    for key in required_keys:
        if key not in logs:
            raise ValueError(f"Training log missing required key: {key}")
    
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Training and Validation Loss
    axs[0, 0].plot(logs['epochs'], logs['train_loss'], 'b-', label='Training Loss')
    axs[0, 0].plot(logs['epochs'], logs['val_loss'], 'r-', label='Validation Loss')
    axs[0, 0].set_title('Training and Validation Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Also create a log scale version for the loss
    axs_log = axs[0, 0].twinx()
    axs_log.semilogy(logs['epochs'], logs['train_loss'], 'b--', alpha=0.3, label='Train Loss (log)')
    axs_log.semilogy(logs['epochs'], logs['val_loss'], 'r--', alpha=0.3, label='Val Loss (log)')
    axs_log.set_ylabel('Loss (log scale)')
    
    # Plot 2: Validation Accuracy
    axs[0, 1].plot(logs['epochs'], logs['val_accuracy'], 'g-')
    axs[0, 1].set_title('Validation Accuracy')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].grid(True)
    
    # Highlight the epoch with the best accuracy
    best_acc_idx = np.argmax(logs['val_accuracy'])
    best_acc = logs['val_accuracy'][best_acc_idx]
    best_acc_epoch = logs['epochs'][best_acc_idx]
    axs[0, 1].plot(best_acc_epoch, best_acc, 'go', markersize=10)
    axs[0, 1].annotate(f'Best: {best_acc:.4f}',
                      xy=(best_acc_epoch, best_acc),
                      xytext=(best_acc_epoch + 1, best_acc),
                      arrowprops=dict(facecolor='black', shrink=0.05))
    
    # Plot 3: Weight MAE
    axs[1, 0].plot(logs['epochs'], logs['weight_mae'], 'm-')
    axs[1, 0].set_title('Weight Mean Absolute Error (MAE)')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('MAE (grams)')
    axs[1, 0].grid(True)
    
    # Highlight the epoch with the best MAE
    best_mae_idx = np.argmin(logs['weight_mae'])
    best_mae = logs['weight_mae'][best_mae_idx]
    best_mae_epoch = logs['epochs'][best_mae_idx]
    axs[1, 0].plot(best_mae_epoch, best_mae, 'mo', markersize=10)
    axs[1, 0].annotate(f'Best: {best_mae:.2f}g',
                      xy=(best_mae_epoch, best_mae),
                      xytext=(best_mae_epoch + 1, best_mae),
                      arrowprops=dict(facecolor='black', shrink=0.05))
    
    # Plot 4: Learning Rate
    if 'learning_rates' in logs:
        axs[1, 1].plot(logs['epochs'], logs['learning_rates'], 'c-')
        axs[1, 1].set_title('Learning Rate Schedule')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Learning Rate')
        axs[1, 1].grid(True)
        
        # Use log scale for learning rate
        axs[1, 1].set_yscale('log')
    else:
        axs[1, 1].text(0.5, 0.5, 'Learning rate data not available',
                      horizontalalignment='center',
                      verticalalignment='center',
                      transform=axs[1, 1].transAxes)
    
    # Add a super title
    fig.suptitle('Training Metrics Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the super title
    
    # Save the figure
    output_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(output_path, dpi=300)
    print(f"Training metrics plot saved to {output_path}")
    
    # Create a second figure focusing on the relationship between learning rate and loss
    if 'learning_rates' in logs:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        # Create a scatter plot of learning rate vs. validation loss
        scatter = ax2.scatter(logs['learning_rates'], logs['val_loss'], 
                             c=logs['epochs'], cmap='viridis',
                             alpha=0.8, s=50)
        
        # Add colorbar to show the epoch progression
        cbar = plt.colorbar(scatter)
        cbar.set_label('Epoch')
        
        ax2.set_xlabel('Learning Rate')
        ax2.set_ylabel('Validation Loss')
        ax2.set_title('Validation Loss vs. Learning Rate')
        ax2.grid(True)
        
        # Use log scale for both axes
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        # Save the second figure
        output_path2 = os.path.join(output_dir, 'lr_vs_loss.png')
        plt.savefig(output_path2, dpi=300)
        print(f"Learning rate vs. loss plot saved to {output_path2}")
    
    # Create a third figure showing validation accuracy and weight MAE over epochs
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    
    # First axis for accuracy
    ax3.plot(logs['epochs'], logs['val_accuracy'], 'g-', label='Validation Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy', color='g')
    ax3.tick_params(axis='y', labelcolor='g')
    ax3.grid(True)
    
    # Second axis for MAE
    ax3_2 = ax3.twinx()
    ax3_2.plot(logs['epochs'], logs['weight_mae'], 'm-', label='Weight MAE')
    ax3_2.set_ylabel('MAE (grams)', color='m')
    ax3_2.tick_params(axis='y', labelcolor='m')
    
    # Add title
    plt.title('Validation Accuracy and Weight MAE Over Training')
    
    # Add combined legend
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_2.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    # Save the third figure
    output_path3 = os.path.join(output_dir, 'accuracy_and_mae.png')
    plt.savefig(output_path3, dpi=300)
    print(f"Accuracy and MAE plot saved to {output_path3}")
    
    print(f"All plots saved to {output_dir}")
    return logs

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize training metrics')
    parser.add_argument('--log_path', type=str, required=True,
                        help='Path to the training log JSON file')
    parser.add_argument('--output_dir', type=str, default='plots',
                        help='Directory to save the output plots')
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    plot_training_metrics(args.log_path, args.output_dir)

if __name__ == '__main__':
    main()
