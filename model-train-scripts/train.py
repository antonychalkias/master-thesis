#!/usr/bin/env python3
"""
Training script for food recognition and weight estimation model.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, OneCycleLR
from sklearn.metrics import accuracy_score, mean_absolute_error
import argparse
import matplotlib.pyplot as plt

from data import prepare_data
from model import MultiTaskNet
from utils import parse_args, setup_paths, get_device
from lr_finder import LRFinder
from one_cycle_lr import OneCycleLR as CustomOneCycleLR

def train_model(model, train_dataloader, val_dataloader, device, num_epochs, model_save_dir, lr_strategy='one_cycle'):
    """
    Train the model and save the best checkpoint
    
    Args:
        model: Model to train
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        device: Device to use for training
        num_epochs: Number of epochs to train for
        model_save_dir: Directory to save the model
        lr_strategy: Learning rate strategy ('one_cycle', 'cosine', 'step', or 'find')
    
    Returns:
        training_logs: Dictionary containing training metrics
    """
    # Loss functions
    criterion_class = nn.CrossEntropyLoss()
    criterion_weight = nn.MSELoss()
    
    # Create plot directory if needed
    plot_dir = os.path.join(model_save_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Determine the best learning rate if requested
    if lr_strategy == 'find':
        print("Running learning rate finder...")
        lr_finder = LRFinder(
            model=model,
            train_dataloader=train_dataloader,
            criterion_class=criterion_class,
            criterion_weight=criterion_weight,
            device=device,
            start_lr=1e-7,
            end_lr=1.0,
            num_iterations=100,
            weight_class=0.7
        )
        lr_finder.find()
        plot_path = os.path.join(plot_dir, "lr_finder_results.png")
        lr_finder.plot(save_path=plot_path)
        
        best_lr = lr_finder.best_lr
        print(f"Learning rate finder complete. Using LR: {best_lr}")
    else:
        # Use the provided learning rate
        best_lr = args.lr
    
    # Optimizer with potentially updated learning rate
    optimizer = optim.Adam(model.parameters(), lr=best_lr, weight_decay=1e-5)
    
    # Choose appropriate learning rate scheduler based on strategy
    steps_per_epoch = len(train_dataloader)
    
    if lr_strategy == 'one_cycle':
        # Built-in One Cycle LR
        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=best_lr * 10,  # Peak LR will be 10x the base LR
            steps_per_epoch=steps_per_epoch,
            epochs=num_epochs,
            pct_start=0.3,  # Spend 30% of iterations in the increasing phase
            div_factor=25,  # initial_lr = max_lr/25
            final_div_factor=1e4,  # min_lr = initial_lr/10000
            anneal_strategy='cos'  # Use cosine annealing
        )
        scheduler_step_on_batch = True
        print(f"Using OneCycleLR scheduler with max LR: {best_lr * 10}")
    elif lr_strategy == 'cosine':
        # Cosine annealing
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=best_lr / 100
        )
        scheduler_step_on_batch = False
        print(f"Using CosineAnnealingLR scheduler with initial LR: {best_lr}")
    else:  # default to 'step'
        # Step LR - more gradual than the original
        scheduler = StepLR(
            optimizer,
            step_size=5,  # Change every 5 epochs instead of 10
            gamma=0.75  # Reduce by 25% instead of 50%
        )
        scheduler_step_on_batch = False
        print(f"Using StepLR scheduler with initial LR: {best_lr}, step_size=5, gamma=0.75")
    
    # Metrics tracking
    best_val_loss = float('inf')
    best_val_accuracy = 0
    best_val_mae = float('inf')
    training_logs = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "weight_mae": [],
        "learning_rates": []  # Track learning rates across epochs
    }

    print(f"Optimizer: Adam with initial learning rate {best_lr}, weight decay 1e-5")
    print(f"Starting training for {num_epochs} epochs...")
    
    # Flag for whether to step scheduler per batch or per epoch
    step_per_batch = (lr_strategy == 'one_cycle')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        
        for images, labels, weights in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            weights = weights.to(device)

            optimizer.zero_grad()
            
            outputs_class, outputs_weight = model(images)
            
            loss_class = criterion_class(outputs_class, labels)
            loss_weight = criterion_weight(outputs_weight, weights)
            
            # Weighted loss - emphasizes classification a bit more than regression
            total_loss = 0.7 * loss_class + 0.3 * loss_weight

            total_loss.backward()
            optimizer.step()
            
            # Step the scheduler if using OneCycleLR
            if lr_strategy == 'one_cycle':
                scheduler.step()
            
            running_train_loss += total_loss.item()
        
        avg_train_loss = running_train_loss / len(train_dataloader)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        all_preds = []
        all_labels = []
        all_weight_preds = []
        all_weight_true = []
        
        with torch.no_grad():
            for images, labels, weights in val_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                weights = weights.to(device)
                
                outputs_class, outputs_weight = model(images)
                
                loss_class = criterion_class(outputs_class, labels)
                loss_weight = criterion_weight(outputs_weight, weights)
                
                total_loss = loss_class + loss_weight
                running_val_loss += total_loss.item()
                
                _, predicted = torch.max(outputs_class, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_weight_preds.extend(outputs_weight.cpu().numpy())
                all_weight_true.extend(weights.cpu().numpy())
        
        avg_val_loss = running_val_loss / len(val_dataloader)
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_mae = mean_absolute_error(all_weight_true, all_weight_preds)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{num_epochs}: LR = {current_lr:.6f}, Train Loss = {avg_train_loss:.4f}, "
              f"Val Loss = {avg_val_loss:.4f}, Val Acc = {val_accuracy:.4f}, Weight MAE = {val_mae:.2f}g")
        
        # Save epoch metrics to log
        training_logs["epochs"].append(epoch + 1)
        training_logs["train_loss"].append(avg_train_loss)
        training_logs["val_loss"].append(avg_val_loss)
        training_logs["val_accuracy"].append(val_accuracy)
        training_logs["weight_mae"].append(val_mae)
        training_logs["learning_rates"].append(current_lr)

        # Check for existing model before saving
        model_path = os.path.join(model_save_dir, "best_model.pth")
        previous_val_loss = float('inf')
        previous_val_accuracy = 0
        previous_val_mae = float('inf')
        
        # Check if model exists and load its metrics
        if os.path.exists(model_path):
            try:
                # Add numpy.core.multiarray.scalar to the safe globals list for PyTorch 2.6+ compatibility
                torch.serialization.add_safe_globals(['numpy.core.multiarray.scalar'])
                # Use weights_only=False for backward compatibility
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
                previous_val_loss = checkpoint.get('val_loss', float('inf'))
                previous_val_accuracy = checkpoint.get('val_accuracy', 0)
                previous_val_mae = checkpoint.get('val_mae', float('inf'))
                
                print("Found existing model with val_loss={:.4f}, val_accuracy={:.4f}, val_mae={:.2f}g".format(
                    previous_val_loss, previous_val_accuracy, previous_val_mae))
            except Exception as e:
                print("Warning: Could not load existing model metrics: {}".format(e))
                # Continue with default values if model can't be loaded
        
        # Determine if current model is better than best seen so far
        save_model = False
        
        # First check if there's an existing saved model and compare with it
        if os.path.exists(model_path):                # Compare current model against the previously saved model (not just this run's best)
            # Only save if current model is better than the previously saved one
            if val_accuracy > previous_val_accuracy:
                save_model = True
                best_val_accuracy = max(best_val_accuracy, val_accuracy)
                print("Better accuracy than saved model: {:.4f} vs {:.4f}".format(val_accuracy, previous_val_accuracy))
            elif val_accuracy >= previous_val_accuracy * 0.95:  # Within 5% of saved model's accuracy
                if avg_val_loss < previous_val_loss * 0.9:
                    save_model = True
                    best_val_loss = min(best_val_loss, avg_val_loss)
                    print("Similar accuracy with significantly better loss: {:.4f} vs {:.4f}".format(avg_val_loss, previous_val_loss))
                elif val_mae < previous_val_mae * 0.9:
                    save_model = True
                    best_val_mae = min(best_val_mae, val_mae)
                    print("Similar accuracy with significantly better MAE: {:.2f}g vs {:.2f}g".format(val_mae, previous_val_mae))
                else:
                    save_model = False
                    print("No significant improvements over saved model - not saving")
            else:
                save_model = False
                print("Worse accuracy than saved model ({:.4f} vs {:.4f}) - not saving".format(val_accuracy, previous_val_accuracy))
        else:
            # No existing model, compare with best metrics from current training run
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                save_model = True
                print("New best accuracy: {:.4f}".format(val_accuracy))
            elif val_accuracy >= best_val_accuracy * 0.98:  # Within 2% of best accuracy
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    save_model = True
                    print("Similar accuracy with better loss: {:.4f} (vs {:.4f})".format(avg_val_loss, best_val_loss))
                elif val_mae < best_val_mae:
                    best_val_mae = val_mae
                    save_model = True
                    print("Similar accuracy with better MAE: {:.2f}g (vs {:.2f}g)".format(val_mae, best_val_mae))
        
        # Special handling for first epoch - we no longer need this since we're properly
        # comparing against the existing saved model already
        if epoch == 0 and os.path.exists(model_path):
            print("First epoch complete - saved model metrics: Acc={:.4f}, Loss={:.4f}, MAE={:.2f}g".format(
                previous_val_accuracy, previous_val_loss, previous_val_mae))
            
        if save_model:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'val_mae': val_mae,
                'label_to_idx': label_to_idx
            }, model_path)
            print("Model saved to {} (Val Loss: {:.4f}, Val Acc: {:.4f}, Val MAE: {:.2f}g)".format(
                model_path, avg_val_loss, val_accuracy, val_mae))
        
        # Step the scheduler for epoch-based schedulers
        if lr_strategy != 'one_cycle':
            scheduler.step()
    
    # Plot the learning rate schedule
    plt.figure(figsize=(10, 4))
    plt.plot(training_logs["epochs"], training_logs["learning_rates"])
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'learning_rate_schedule.png'), dpi=300)

    return training_logs

if __name__ == '__main__':
    # Required for multiprocessing on macOS
    mp.set_start_method('spawn', force=True)
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train food recognition and weight estimation model")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to the CSV file with annotations")
    parser.add_argument("--images_dir", type=str, default=None, help="Path to the directory with images")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")  # Changed default to 5e-5
    parser.add_argument("--model_dir", type=str, default=None, help="Directory to save the model")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--lr_strategy", type=str, default="one_cycle",
                        choices=["one_cycle", "cosine", "step", "find"],
                        help="Learning rate strategy to use")
    args = parser.parse_args()
    args = setup_paths(args)
    
    # Prepare data
    train_dataloader, val_dataloader, label_to_idx = prepare_data(
        args.csv_path, 
        args.images_dir, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    
    # Get device (CUDA, MPS, or CPU)
    device = get_device()
    
    # Initialize model
    num_classes = len(label_to_idx)
    model = MultiTaskNet(num_classes)
    model.to(device)
    
    # Train the model
    training_logs = train_model(
        model, 
        train_dataloader, 
        val_dataloader, 
        device, 
        args.epochs, 
        args.model_dir,
        lr_strategy=args.lr_strategy
    )
    
    # Save training log as JSON
    # Convert any numpy values to Python native types for JSON serialization
    json_compatible_logs = {
        "epochs": [int(x) for x in training_logs["epochs"]],
        "train_loss": [float(x) for x in training_logs["train_loss"]],
        "val_loss": [float(x) for x in training_logs["val_loss"]],
        "val_accuracy": [float(x) for x in training_logs["val_accuracy"]],
        "weight_mae": [float(x) for x in training_logs["weight_mae"]],
        "learning_rates": [float(x) for x in training_logs["learning_rates"]]
    }
    
    log_path = os.path.join(args.model_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(json_compatible_logs, f, indent=4)
    print("Training log saved to {}".format(log_path))
    
    print("Training completed!")
