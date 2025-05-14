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
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, mean_absolute_error

from data import prepare_data
from model import MultiTaskNet
from utils import parse_args, setup_paths, get_device

def train_model(model, train_dataloader, val_dataloader, device, num_epochs, model_save_dir):
    """
    Train the model and save the best checkpoint
    
    Args:
        model: Model to train
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        device: Device to use for training
        num_epochs: Number of epochs to train for
        model_save_dir: Directory to save the model
    
    Returns:
        training_logs: Dictionary containing training metrics
    """
    # Loss functions
    criterion_class = nn.CrossEntropyLoss()
    criterion_weight = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Metrics tracking
    best_val_loss = float('inf')
    best_val_accuracy = 0
    best_val_mae = float('inf')
    training_logs = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "weight_mae": []
    }

    print(f"Optimizer: Adam with learning rate {args.lr}, weight decay 1e-5")
    print(f"Starting training for {num_epochs} epochs...")

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
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, "
              f"Val Acc = {val_accuracy:.4f}, Weight MAE = {val_mae:.2f}g")
        
        # Save epoch metrics to log
        training_logs["epochs"].append(epoch + 1)
        training_logs["train_loss"].append(avg_train_loss)
        training_logs["val_loss"].append(avg_val_loss)
        training_logs["val_accuracy"].append(val_accuracy)
        training_logs["weight_mae"].append(val_mae)

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
        
        scheduler.step()

    return training_logs

if __name__ == '__main__':
    # Required for multiprocessing on macOS
    mp.set_start_method('spawn', force=True)
    
    # Parse command-line arguments
    args = parse_args()
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
        args.model_dir
    )
    
    # Save training log as JSON
    # Convert any numpy values to Python native types for JSON serialization
    json_compatible_logs = {
        "epochs": [int(x) for x in training_logs["epochs"]],
        "train_loss": [float(x) for x in training_logs["train_loss"]],
        "val_loss": [float(x) for x in training_logs["val_loss"]],
        "val_accuracy": [float(x) for x in training_logs["val_accuracy"]],
        "weight_mae": [float(x) for x in training_logs["weight_mae"]]
    }
    
    log_path = os.path.join(args.model_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(json_compatible_logs, f, indent=4)
    print("Training log saved to {}".format(log_path))
    
    print("Training completed!")
