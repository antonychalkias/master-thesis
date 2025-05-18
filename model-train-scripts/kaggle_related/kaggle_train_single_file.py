#!/usr/bin/env python3
"""
Kaggle-specific training script for food recognition and weight estimation model.
This is an adaptation of the original train.py specifically for the Kaggle environment.
This is a self-contained version with all dependencies included (no separate early_stopping module).
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, OneCycleLR
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
from sklearn.metrics import accuracy_score, mean_absolute_error
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
import time
import sys
from tqdm import tqdm

# ===================== EARLY STOPPING IMPLEMENTATION =====================

class EarlyStopping:
    """
    Early stopping to stop the training when the monitored metric doesn't improve after a given patience.
    Saves the best model when the monitored metric improves.
    
    Args:
        patience (int): How many epochs to wait after last improvement.
                        Default: 10
        verbose (bool): If True, prints a message for each improvement.
                       Default: True
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                       Default: 0.0001
        path (str): Path for model checkpoint saving.
                    Default: 'checkpoint.pt'
        monitor_mode (str): 'min' for metrics that should decrease (e.g., loss),
                          'max' for metrics that should increase (e.g., accuracy).
                          Default: 'max'
        trace_func (callable): Function for logging messages.
                            Default: print
    """
    def __init__(self, patience=10, verbose=True, delta=0.0001, path='checkpoint.pt', 
                 monitor_mode='max', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.monitor_mode = monitor_mode
        self.trace_func = trace_func
        self.improved = False
        self.best_epoch = 0
        self.start_time = time.time()
        
        # Set the direction for comparison based on monitor_mode
        if self.monitor_mode == 'min':
            self.improved_condition = lambda current, best: current < best - self.delta
        elif self.monitor_mode == 'max':
            self.improved_condition = lambda current, best: current > best + self.delta
        else:
            raise ValueError(f"Invalid monitor_mode: {monitor_mode}. Use 'min' or 'max'")
    
    def __call__(self, score, model, optimizer=None, epoch=None, val_loss=None, val_accuracy=None, 
                weight_mae=None, composite_score=None, label_to_idx=None, path=None):
        """
        Check if the training should be stopped.
        
        Args:
            score (float): The score to monitor for improvement
            model (torch.nn.Module): The model to save when improved
            optimizer (torch.optim.Optimizer, optional): The optimizer to save with the model
            epoch (int, optional): Current epoch number
            val_loss (float, optional): Validation loss
            val_accuracy (float, optional): Validation accuracy
            weight_mae (float, optional): Weight MAE
            composite_score (float, optional): Composite score
            label_to_idx (dict, optional): Label to index mapping
            path (str, optional): Path to save checkpoint (overrides self.path)
            
        Returns:
            bool: Whether early stopping should be triggered
        """
        if self.patience <= 0:  # Early stopping disabled
            return False
            
        self.improved = False
        save_path = path if path is not None else self.path
        
        if self.best_score is None:
            # First validation run
            # Check if there's a previous best model to compare with
            device = next(model.parameters()).device
            if _compare_with_previous_best(score, save_path, device):
                self.best_score = score
                self.improved = True
                self.best_epoch = epoch if epoch is not None else 0
                self.save_checkpoint(model, optimizer, epoch, val_loss, val_accuracy, weight_mae, 
                                   composite_score, label_to_idx, save_path)
            else:
                # Previous model is better, load its score
                previous_checkpoint = _load_best_model_if_exists(save_path, device)
                if previous_checkpoint:
                    self.best_score = previous_checkpoint.get('composite_score', score)
                    self.best_epoch = previous_checkpoint.get('epoch', 0) - 1  # Adjust for 0-indexing
                    self.trace_func(f'Keeping previous model with better score: {self.best_score:.4f}')
                else:
                    # This should not happen, but just in case
                    self.best_score = score
                    self.improved = True
                    self.save_checkpoint(model, optimizer, epoch, val_loss, val_accuracy, weight_mae, 
                                       composite_score, label_to_idx, save_path)
        elif self.improved_condition(score, self.best_score):
            # Improvement
            if self.verbose:
                metric_name = "loss" if self.monitor_mode == 'min' else "score"
                self.trace_func(f'Validation {metric_name} improved from {self.best_score:.4f} to {score:.4f}')
            self.best_score = score
            self.improved = True
            self.best_epoch = epoch if epoch is not None else 0
            self.counter = 0
            self.save_checkpoint(model, optimizer, epoch, val_loss, val_accuracy, weight_mae, 
                               composite_score, label_to_idx, save_path)
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                if self.verbose:
                    self.trace_func(f'Early stopping triggered after epoch {epoch+1 if epoch is not None else "unknown"}')
                    elapsed_time = time.time() - self.start_time
                    self.trace_func(f'Best performance was at epoch {self.best_epoch+1 if self.best_epoch is not None else "unknown"} with {self.monitor_mode} value: {self.best_score:.4f}')
                    self.trace_func(f'Training ran for {elapsed_time/60:.2f} minutes before early stopping')
                self.early_stop = True
        
        return self.early_stop
    
    def save_checkpoint(self, model, optimizer=None, epoch=None, val_loss=None, val_accuracy=None, 
                      weight_mae=None, composite_score=None, label_to_idx=None, path=None):
        """Save model checkpoint when validation metric improves."""
        if self.verbose:
            self.trace_func(f'Saving model checkpoint to {path if path else self.path}')
        
        save_path = path if path is not None else self.path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # Prepare checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'best_score': self.best_score,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if epoch is not None:
            checkpoint['epoch'] = epoch + 1
            
        if val_loss is not None:
            checkpoint['val_loss'] = val_loss
            
        if val_accuracy is not None:
            checkpoint['val_accuracy'] = val_accuracy
            
        if weight_mae is not None:
            checkpoint['val_mae'] = weight_mae
            
        if composite_score is not None:
            checkpoint['composite_score'] = composite_score
            
        if label_to_idx is not None:
            checkpoint['label_to_idx'] = label_to_idx
        
        # Save checkpoint
        torch.save(checkpoint, save_path)

# ===================== MODEL DEFINITION =====================

class MultiTaskNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load pretrained EfficientNet-B0
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # Get the number of features before the classification layer
        num_features = self.backbone.classifier[1].in_features

        # Remove original classifier head
        self.backbone.classifier = nn.Identity()

        # Add multitask heads
        self.classifier = nn.Linear(num_features, num_classes)
        self.weight_regressor = nn.Linear(num_features, 1)

    def forward(self, x):
        # Extract features from the backbone
        features = self.backbone(x)
        
        # Pass features through task-specific heads
        class_logits = self.classifier(features)
        weight_pred = self.weight_regressor(features)
        
        return class_logits, weight_pred

# ===================== DATA HANDLING =====================

def fix_csv_file(csv_path, output_path=None):
    """Fixes CSV file with potential delimiter issues"""
    try:
        # Try to peek at the file to determine its format
        with open(csv_path, 'r') as f:
            first_line = f.readline().strip()
        print(f"First line of CSV: {first_line}")
        
        # Decide on delimiter
        if ';' in first_line:
            print("Detected semicolon delimiter, loading with sep=';'")
            df = pd.read_csv(csv_path, sep=';')
        else:
            print("Using default comma delimiter")
            df = pd.read_csv(csv_path)
            
        # Check if we still have delimiter in column names
        if len(df.columns) == 1 and (';' in df.columns[0] or ',' in df.columns[0]):
            print(f"Single column with delimiters detected: {df.columns[0]}")
            
            # Try manual parsing
            with open(csv_path, 'r') as f:
                lines = [line.strip() for line in f.readlines()]
            
            # Determine delimiter
            delim = ';' if ';' in first_line else ','
            data = [line.split(delim) for line in lines]
            headers = data[0]
            rows = data[1:]
            
            df = pd.DataFrame(rows, columns=headers)
            print(f"Manual parsing successful. Columns: {df.columns.tolist()}")
        
        print(f"CSV loaded successfully with columns: {df.columns.tolist()}")
        print(f"Sample data:\n{df.head(2)}")
        
        # Save fixed CSV if output path provided
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Fixed CSV saved to: {output_path}")
            
        return df
    except Exception as e:
        print(f"Error fixing CSV: {e}")
        raise

class FoodDataset(Dataset):
    def __init__(self, data_df, images_dir, transform=None):
        """
        Initialize dataset with a pre-loaded dataframe
        
        Args:
            data_df: Pandas DataFrame with image paths and labels
            images_dir: Directory containing the images
            transform: Transforms to apply to the images
        """
        self.data = data_df
        self.images_dir = images_dir
        self.transform = transform
        
        # Detect column names
        img_col_options = ['image_name', 'filename', 'image']
        self.img_col = next((col for col in img_col_options if col in self.data.columns), self.data.columns[0])
        
        label_col_options = ['label', 'labels', 'class', 'food']
        self.label_col = next((col for col in label_col_options if col in self.data.columns), None)
        if not self.label_col:
            # If no matching column, use the second column (typically labels)
            if len(self.data.columns) > 1:
                self.label_col = self.data.columns[1]
            else:
                raise ValueError(f"Could not find label column in {self.data.columns.tolist()}")
        
        weight_col_options = ['weight', 'weights']
        self.weight_col = next((col for col in weight_col_options if col in self.data.columns), None)
        if not self.weight_col:
            # Try to find a numeric column
            numeric_cols = self.data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                self.weight_col = numeric_cols[0]
            else:
                raise ValueError(f"Could not find weight column in {self.data.columns.tolist()}")
        
        print(f"Using columns - Image: {self.img_col}, Label: {self.label_col}, Weight: {self.weight_col}")
        
        # Create mapping from class name to idx
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(self.data[self.label_col].unique()))}
        self.num_classes = len(self.label_to_idx)
        print(f"Found {self.num_classes} unique food classes")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row[self.img_col]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Get weight (ensure numeric)
        weight = float(row[self.weight_col])
        
        # Get class label
        label = row[self.label_col]
        label_idx = self.label_to_idx[label]
        
        # Apply transformations
        if self.transform:
            img = self.transform(img)
            
        return img, label_idx, weight

def prepare_data(csv_path, images_dir, batch_size=16, num_workers=0, val_split=0.2):
    """
    Prepare train and validation dataloaders
    
    Args:
        csv_path: Path to the CSV file with annotations
        images_dir: Path to the directory with images
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        val_split: Fraction of data to use for validation
        
    Returns:
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        label_to_idx: Dictionary mapping class names to indices
        full_dataset: The full dataset instance with the label_to_idx attribute
    """
    # Fix and load CSV file
    fixed_csv = fix_csv_file(csv_path)
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    full_dataset = FoodDataset(fixed_csv, images_dir, transform)
    
    # Create train/val split
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training on {train_size} samples, validating on {val_size} samples")
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_dataloader, val_dataloader, full_dataset.label_to_idx, full_dataset

# ===================== TRAINING UTILITIES =====================

def get_device():
    """Get available device (CUDA or CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def train_model(model, train_dataloader, val_dataloader, device, num_epochs, model_dir, full_dataset=None, 
               lr_strategy="one_cycle", best_lr=5e-5, early_stopping_patience=10, reduce_lr_patience=5):
    """
    Train the multi-task model for food recognition and weight estimation
    
    Args:
        model: The model to train
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        device: Device to use for training
        num_epochs: Number of epochs to train
        model_dir: Directory to save the model
        full_dataset: The full dataset with label_to_idx attribute
        lr_strategy: Learning rate strategy to use ("one_cycle", "cosine", "step")
        best_lr: Base learning rate to use
        early_stopping_patience: Number of epochs to wait before early stopping
        reduce_lr_patience: Number of epochs to wait before reducing learning rate
    
    Returns:
        Dictionary containing training logs
    """
    print(f"Training model for {num_epochs} epochs with {lr_strategy} learning rate schedule")
    print(f"Early stopping patience: {early_stopping_patience}, Reduce LR patience: {reduce_lr_patience}")
    
    # Initialize optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=best_lr, weight_decay=1e-4)
    
    # Initialize learning rate scheduler based on strategy
    base_scheduler = _create_scheduler(optimizer, lr_strategy, num_epochs, len(train_dataloader), best_lr)
    
    # Add ReduceLROnPlateau scheduler for adaptive learning rate reduction
    # This will be used in addition to the base scheduler for strategies other than one_cycle
    reduce_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=reduce_lr_patience, verbose=True, min_lr=1e-7
    )
    
    # Loss functions
    class_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.SmoothL1Loss()
    
    # Initialize training logs and tracking variables
    # Use the label_to_idx from the full dataset
    label_to_idx = full_dataset.label_to_idx if full_dataset else getattr(val_dataloader.dataset, 'label_to_idx', {})
    training_logs = _initialize_training_logs(label_to_idx)
    
    # For saving best model
    best_model_path = os.path.join(model_dir, "best_model.pth")
    final_model_path = os.path.join(model_dir, "final_model.pth")
    
    # Initialize early stopping - we monitor the composite score which we want to maximize
    early_stopping = EarlyStopping(patience=early_stopping_patience, monitor_mode='max', verbose=True)
    
    # Training start time
    start_time = time.time()
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        train_loss, train_accuracy = _train_epoch(model, train_dataloader, optimizer, class_criterion, 
                                                regression_criterion, device, epoch, num_epochs, 
                                                lr_strategy, base_scheduler)
        
        # Validation phase
        val_metrics = _validate_epoch(model, val_dataloader, class_criterion, 
                                     regression_criterion, device, epoch, num_epochs)
        
        # Extract validation metrics
        val_loss = val_metrics['val_loss']
        val_accuracy = val_metrics['val_accuracy']
        weight_mae = val_metrics['weight_mae']
        
        # Update learning rate for schedulers that step per epoch
        if lr_strategy in ["cosine", "step"]:
            base_scheduler.step()
        
        # For LR reduction on plateau, update based on validation loss
        if lr_strategy != "one_cycle":
            reduce_lr_scheduler.step(val_loss)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        _log_epoch_metrics(epoch, num_epochs, train_loss, train_accuracy, val_loss, val_accuracy, weight_mae, current_lr)
        
        # Update training logs
        _update_training_logs(training_logs, epoch, train_loss, train_accuracy, val_loss, val_accuracy, 
                             weight_mae, current_lr, val_metrics)
        
        # Calculate composite score (accuracy - mae/100)
        composite_score = val_accuracy - (weight_mae / 100)
        
        # Early stopping check (using composite score)
        if early_stopping(composite_score, model, optimizer, epoch, val_loss, val_accuracy, 
                          weight_mae, composite_score, label_to_idx, best_model_path):
            print(f"Early stopping triggered after {epoch+1} epochs")
            # Load the best model for the return
            best_checkpoint = _load_best_model_if_exists(best_model_path, device)
            if best_checkpoint:
                model.load_state_dict(best_checkpoint['model_state_dict'])
                print(f"Loaded best model from epoch {best_checkpoint.get('epoch', 0)}")
            break
    
    # Training complete
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes")
    
    # Calculate final composite score 
    final_accuracy = training_logs["val_accuracy"][-1]
    final_mae = training_logs["weight_mae"][-1]
    final_score = final_accuracy - (final_mae / 100)
    
    # Log best performance
    best_score = early_stopping.best_score
    print(f"Best composite score: {best_score:.4f}")
    
    # Save the final model with comprehensive metrics
    _save_final_model(model, optimizer, epoch+1, training_logs, final_accuracy, final_mae, 
                     final_score, label_to_idx, total_time, final_model_path)
    
    # Ensure we're using the best model
    if best_score > final_score:
        print(f"Note: Best model ({best_score:.4f}) is better than final model ({final_score:.4f})")
        print(f"Best model was saved to {best_model_path}")
    
    return training_logs

def _create_scheduler(optimizer, lr_strategy, num_epochs, steps_per_epoch, best_lr):
    """Helper function to create the learning rate scheduler"""
    if lr_strategy == "one_cycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=best_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=num_epochs,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1000.0
        )
    elif lr_strategy == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=num_epochs
        )
    elif lr_strategy == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=5,
            gamma=0.1
        )
    else:
        raise ValueError(f"Unknown learning rate strategy: {lr_strategy}")

def _initialize_training_logs(label_to_idx):
    """Initialize the training logs dictionary with all required keys"""
    training_logs = {
        "epochs": [],
        "train_loss": [],
        "train_accuracy": [],  # Added train_accuracy
        "val_loss": [],
        "val_accuracy": [],
        "weight_mae": [],
        "lr": [],
        "per_class_precision": {},
        "per_class_recall": {},
        "per_class_f1": {},
        "per_class_mae": {},
        "confusion_matrix": []
    }
    
    # Initialize per-class metrics
    label_map = {idx: label for label, idx in label_to_idx.items()}
    for idx, label in label_map.items():
        training_logs["per_class_precision"][label] = []
        training_logs["per_class_recall"][label] = []
        training_logs["per_class_f1"][label] = []
        training_logs["per_class_mae"][label] = []
    
    return training_logs

def _train_epoch(model, train_dataloader, optimizer, class_criterion, regression_criterion, 
               device, epoch, num_epochs, lr_strategy, scheduler):
    """Run one training epoch and return the average training loss and accuracy"""
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar for training
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    
    for images, labels, weights in pbar:
        images = images.to(device)
        labels = labels.to(device)
        weights = weights.to(device)
        
        # Forward pass
        label_preds, weight_preds = model(images)
        
        # Calculate losses
        class_loss = class_criterion(label_preds, labels)
        reg_loss = regression_criterion(weight_preds.squeeze(), weights)
        
        # Combined loss
        loss = class_loss + reg_loss
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update learning rate if using one-cycle policy
        if lr_strategy == "one_cycle":
            scheduler.step()
        
        # Calculate training accuracy
        _, predicted = torch.max(label_preds.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update training loss
        train_loss += loss.item()
        
        # Calculate current training accuracy
        train_accuracy = 100 * correct / total if total > 0 else 0
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'class_loss': class_loss.item(),
            'reg_loss': reg_loss.item(),
            'accuracy': f'{train_accuracy:.2f}%'
        })
    
    # Calculate average training loss and accuracy for the epoch
    train_loss /= len(train_dataloader)
    train_accuracy = correct / total if total > 0 else 0
    
    return train_loss, train_accuracy

def _validate_epoch(model, val_dataloader, class_criterion, regression_criterion, 
                  device, epoch, num_epochs):
    """Run one validation epoch and return metrics"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    weight_error = 0.0
    
    # Get the mapping of label indices to names
    # Access the dataset to get the full dataset's label_to_idx attribute
    full_dataset = val_dataloader.dataset.dataset if hasattr(val_dataloader.dataset, 'dataset') else val_dataloader.dataset
    label_to_idx = getattr(full_dataset, 'label_to_idx', {})
    label_map = {idx: label for label, idx in label_to_idx.items()}
    num_classes = len(label_map)
    
    # Initialize per-class metrics
    class_correct = {idx: 0 for idx in label_map.keys()}
    class_total = {idx: 0 for idx in label_map.keys()}
    class_weight_error = {idx: 0.0 for idx in label_map.keys()}
    class_weight_samples = {idx: 0 for idx in label_map.keys()}
    
    # For confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    # For precision, recall calculation
    true_positives = {idx: 0 for idx in label_map.keys()}
    false_positives = {idx: 0 for idx in label_map.keys()}
    false_negatives = {idx: 0 for idx in label_map.keys()}
    
    # Progress bar for validation
    pbar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
    
    with torch.no_grad():
        for images, labels, weights in pbar:
            images = images.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            
            # Forward pass
            label_preds, weight_preds = model(images)
            
            # Calculate losses
            class_loss = class_criterion(label_preds, labels)
            reg_loss = regression_criterion(weight_preds.squeeze(), weights)
            loss = class_loss + reg_loss
            
            # Update validation loss
            val_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(label_preds, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Calculate weight estimation error (MAE)
            weight_error += torch.abs(weight_preds.squeeze() - weights).sum().item()
            
            # Update per-class metrics and confusion matrix
            _update_class_metrics(labels, predicted, weight_preds, weights, class_total, 
                                class_correct, class_weight_error, class_weight_samples,
                                true_positives, false_positives, false_negatives, confusion)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': (predicted == labels).sum().item() / labels.size(0)
            })
    
    # Calculate average validation metrics
    val_loss /= len(val_dataloader)
    val_accuracy = correct / total
    weight_mae = weight_error / total
    
    # Calculate per-class metrics
    per_class_metrics = _calculate_per_class_metrics(class_total, class_weight_error, 
                                                  class_weight_samples, true_positives, 
                                                  false_positives, false_negatives, label_map)
    
    # Combine all metrics
    return {
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'weight_mae': weight_mae,
        'confusion_matrix': confusion.tolist(),
        'per_class_metrics': per_class_metrics
    }

def _update_class_metrics(labels, predicted, weight_preds, weights, class_total, class_correct, 
                        class_weight_error, class_weight_samples, true_positives, false_positives, 
                        false_negatives, confusion):
    """Update per-class metrics based on prediction results"""
    for i in range(len(labels)):
        true_label = labels[i].item()
        pred_label = predicted[i].item()
        
        # Update confusion matrix
        confusion[true_label][pred_label] += 1
        
        # Update class statistics
        class_total[true_label] += 1
        
        if true_label == pred_label:
            class_correct[true_label] += 1
            true_positives[true_label] += 1
        else:
            false_positives[pred_label] += 1
            false_negatives[true_label] += 1
        
        # Update weight error per class
        class_weight_error[true_label] += torch.abs(weight_preds[i] - weights[i]).item()
        class_weight_samples[true_label] += 1

def _calculate_per_class_metrics(class_total, class_weight_error, class_weight_samples, 
                               true_positives, false_positives, false_negatives, label_map):
    """Calculate precision, recall, F1, and MAE for each class"""
    per_class_metrics = {
        'precision': {},
        'recall': {},
        'f1': {},
        'mae': {}
    }
    
    for idx, label in label_map.items():
        # Avoid division by zero
        if class_total[idx] > 0:
            precision = true_positives[idx] / max(1, (true_positives[idx] + false_positives[idx]))
            recall = true_positives[idx] / max(1, (true_positives[idx] + false_negatives[idx]))
            f1 = 2 * (precision * recall) / max(1e-8, (precision + recall))
            mae = class_weight_error[idx] / max(1, class_weight_samples[idx])
        else:
            precision, recall, f1, mae = 0.0, 0.0, 0.0, 0.0
        
        per_class_metrics['precision'][label] = float(precision)
        per_class_metrics['recall'][label] = float(recall)
        per_class_metrics['f1'][label] = float(f1)
        per_class_metrics['mae'][label] = float(mae)
    
    return per_class_metrics

def _log_epoch_metrics(epoch, num_epochs, train_loss, train_accuracy, val_loss, val_accuracy, weight_mae, current_lr):
    """Log the metrics for the current epoch"""
    print(f"Epoch {epoch+1}/{num_epochs} completed")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    print(f"Weight MAE: {weight_mae:.4f}, Learning Rate: {current_lr:.8f}")

def _update_training_logs(training_logs, epoch, train_loss, train_accuracy, val_loss, val_accuracy, weight_mae, 
                        current_lr, val_metrics):
    """Update the training logs with metrics from the current epoch"""
    # Update basic metrics
    training_logs["epochs"].append(epoch + 1)
    training_logs["train_loss"].append(train_loss)
    training_logs["train_accuracy"].append(train_accuracy)  # Added train_accuracy
    training_logs["val_loss"].append(val_loss)
    training_logs["val_accuracy"].append(val_accuracy)
    training_logs["weight_mae"].append(weight_mae)
    training_logs["lr"].append(current_lr)
    
    # Update confusion matrix
    training_logs["confusion_matrix"].append(val_metrics['confusion_matrix'])
    
    # Update per-class metrics
    for metric_type in ['precision', 'recall', 'f1', 'mae']:
        for label, value in val_metrics['per_class_metrics'][metric_type].items():
            metric_key = f"per_class_{metric_type}"
            training_logs[metric_key][label].append(value)

def _save_checkpoint(model, optimizer, epoch, val_loss, val_accuracy, weight_mae, 
                   composite_score, label_to_idx, path):
    """Save a model checkpoint"""
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'val_mae': weight_mae,
        'composite_score': composite_score,
        'label_to_idx': label_to_idx
    }, path)

def _save_final_model(model, optimizer, num_epochs, training_logs, final_accuracy, final_mae, 
                    final_score, label_to_idx, total_time, path):
    """Save the final model with comprehensive metrics"""
    # Extract per-class metrics from the last epoch
    per_class_metrics = {
        'precision': {label: values[-1] for label, values in training_logs["per_class_precision"].items()},
        'recall': {label: values[-1] for label, values in training_logs["per_class_recall"].items()},
        'f1': {label: values[-1] for label, values in training_logs["per_class_f1"].items()},
        'mae': {label: values[-1] for label, values in training_logs["per_class_mae"].items()}
    }
    
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': training_logs["val_loss"][-1],
        'val_accuracy': final_accuracy,
        'val_mae': final_mae,
        'composite_score': final_score,
        'label_to_idx': label_to_idx,
        'training_time_minutes': total_time / 60,
        'training_history': {
            'train_loss': training_logs["train_loss"],
            'val_loss': training_logs["val_loss"],
            'val_accuracy': training_logs["val_accuracy"],
            'weight_mae': training_logs["weight_mae"],
            'lr': training_logs["lr"]
        },
        'per_class_metrics': per_class_metrics,
        'final_confusion_matrix': training_logs["confusion_matrix"][-1] if training_logs["confusion_matrix"] else None
    }, path)
    print(f"Final model saved to {path}")
    
def _check_model_improvement(best_model_path, composite_score, best_score, val_accuracy, val_loss, weight_mae):
    """
    Check if the current model performance is better than the previous best model.
    
    Args:
        best_model_path (str): Path where to save the best model
        composite_score (float): Current composite score calculated as val_accuracy - (weight_mae / 100)
        best_score (float): Previous best score to compare with
        val_accuracy (float): Current validation accuracy
        val_loss (float): Current validation loss
        weight_mae (float): Current weight mean absolute error
        
    Returns:
        tuple: (save_model, best_score) - A boolean indicating whether to save the model, 
                                          and the updated best score
    """
    # Check if this is the first evaluation or if the current score is better than the best
    if best_score is None or composite_score > best_score:
        return True, composite_score
    return False, best_score

def _load_best_model_if_exists(model_path, device='cpu'):
    """
    Check if a best model exists and load its metrics for comparison.
    
    Args:
        model_path (str): Path to the model file
        device (str): Device to load the model to
        
    Returns:
        dict or None: The loaded checkpoint if it exists, None otherwise
    """
    if os.path.exists(model_path):
        try:
            # We're just loading to check metrics, so use map_location to avoid CUDA issues
            checkpoint = torch.load(model_path, map_location=device)
            print(f"Found existing model checkpoint at {model_path}:")
            print(f"  - Saved at epoch: {checkpoint.get('epoch', 0)}")
            print(f"  - Validation accuracy: {checkpoint.get('val_accuracy', 0):.4f}")
            print(f"  - Weight MAE: {checkpoint.get('val_mae', 0):.4f}")
            print(f"  - Composite score: {checkpoint.get('composite_score', 0):.4f}")
            return checkpoint
        except Exception as e:
            print(f"Error loading existing model: {str(e)}")
    return None

def _compare_with_previous_best(current_composite_score, model_path, device='cpu'):
    """
    Compare current model performance with the previously saved best model.
    
    Args:
        current_composite_score (float): Composite score of the current model
        model_path (str): Path to the saved best model
        device (str): Device to load the model to
        
    Returns:
        bool: True if current model is better, False otherwise
    """
    # Try to load previous best model
    previous_checkpoint = _load_best_model_if_exists(model_path, device)
    
    if previous_checkpoint is None:
        # No previous model exists, so current is better by default
        return True
    
    # Get previous composite score
    previous_score = previous_checkpoint.get('composite_score', 0)
    
    # Compare scores
    if current_composite_score > previous_score:
        print(f"Current model is better: {current_composite_score:.4f} > {previous_score:.4f}")
        return True
    else:
        print(f"Previous model is better: {previous_score:.4f} >= {current_composite_score:.4f}")
        return False

# ===================== HELPER FUNCTIONS FOR ANALYSIS =====================

def _get_best_class_metrics(training_logs):
    """
    Extract the best metrics for each class across all epochs
    
    Args:
        training_logs (dict): The training logs dictionary
        
    Returns:
        dict: A dictionary with best metrics for each class
    """
    best_metrics = {}
    
    # Get all classes
    all_classes = list(training_logs["per_class_precision"].keys())
    
    for class_name in all_classes:
        # Extract precision history
        precision_history = training_logs["per_class_precision"][class_name]
        recall_history = training_logs["per_class_recall"][class_name]
        f1_history = training_logs["per_class_f1"][class_name]
        mae_history = training_logs["per_class_mae"][class_name]
        
        # Find best F1
        best_f1_idx = np.argmax(f1_history)
        # Find best MAE (lowest)
        best_mae_idx = np.argmin(mae_history)
        
        best_metrics[class_name] = {
            "best_f1": {
                "value": float(f1_history[best_f1_idx]),
                "epoch": int(training_logs["epochs"][best_f1_idx]),
                "precision": float(precision_history[best_f1_idx]),
                "recall": float(recall_history[best_f1_idx])
            },
            "best_mae": {
                "value": float(mae_history[best_mae_idx]),
                "epoch": int(training_logs["epochs"][best_mae_idx])
            },
            "final_values": {
                "precision": float(precision_history[-1]),
                "recall": float(recall_history[-1]),
                "f1": float(f1_history[-1]),
                "mae": float(mae_history[-1])
            }
        }
    
    return best_metrics

def _analyze_convergence(training_logs):
    """
    Analyze the convergence behavior of the training
    
    Args:
        training_logs (dict): The training logs dictionary
        
    Returns:
        dict: Convergence analysis information
    """
    train_loss = training_logs["train_loss"]
    val_loss = training_logs["val_loss"]
    val_acc = training_logs["val_accuracy"]
    weight_mae = training_logs["weight_mae"]
    
    # Determine if training is still improving at the end
    final_window = 5  # Last 5 epochs
    
    # Ensure we have enough epochs
    if len(train_loss) <= final_window:
        final_window = max(1, len(train_loss) // 2)
    
    # Get trend in last few epochs
    train_loss_trend = np.mean(np.diff(train_loss[-final_window:]))
    val_loss_trend = np.mean(np.diff(val_loss[-final_window:]))
    val_acc_trend = np.mean(np.diff(val_acc[-final_window:]))
    weight_mae_trend = np.mean(np.diff(weight_mae[-final_window:]))
    
    # Check for overfitting
    is_overfitting = val_loss_trend > 0 and train_loss_trend < 0
    
    # Determine early stopping point (if validation loss starts to increase)
    early_stop_epoch = None
    if len(val_loss) > 10:  # Only check if we have enough epochs
        # Get the index of minimum val loss after the first few epochs
        min_val_loss_idx = np.argmin(val_loss[5:]) + 5
        
        # Check if val loss keeps increasing after min point
        if min_val_loss_idx < len(val_loss) - 5:
            # If val loss is consistently higher for the next epochs
            if np.mean(val_loss[min_val_loss_idx+1:min_val_loss_idx+6]) > val_loss[min_val_loss_idx]:
                early_stop_epoch = int(training_logs["epochs"][min_val_loss_idx])
    
    # Compute smoothness of convergence
    loss_volatility = np.std(np.diff(val_loss)) / np.mean(val_loss)
    acc_volatility = np.std(np.diff(val_acc)) / max(0.001, np.mean(val_acc))
    mae_volatility = np.std(np.diff(weight_mae)) / max(0.001, np.mean(weight_mae))
    
    return {
        "is_still_improving_at_end": {
            "train_loss": train_loss_trend < 0,
            "val_loss": val_loss_trend < 0,
            "val_accuracy": val_acc_trend > 0,
            "weight_mae": weight_mae_trend < 0
        },
        "convergence_status": {
            "converged": abs(val_loss_trend) < 0.001 and abs(val_acc_trend) < 0.001,
            "overfitting_detected": is_overfitting,
            "potential_early_stop_epoch": early_stop_epoch,
            "validation_loss_volatility": float(loss_volatility),
            "validation_accuracy_volatility": float(acc_volatility),
            "weight_mae_volatility": float(mae_volatility)
        },
        "trends": {
            "final_train_loss_trend": float(train_loss_trend),
            "final_val_loss_trend": float(val_loss_trend),
            "final_val_acc_trend": float(val_acc_trend),
            "final_weight_mae_trend": float(weight_mae_trend)
        }
    }

def _calculate_weight_estimation_stats(training_logs):
    """
    Calculate statistics related to weight estimation performance
    
    Args:
        training_logs (dict): The training logs dictionary
        
    Returns:
        dict: Weight estimation statistics
    """
    # Get weight MAE across all epochs
    weight_mae = training_logs["weight_mae"]
    
    # Get final per-class MAE
    final_class_mae = {class_name: values[-1] 
                      for class_name, values in training_logs["per_class_mae"].items()}
    
    # Calculate statistics
    class_mae_values = list(final_class_mae.values())
    
    return {
        "overall_final_mae": float(weight_mae[-1]),
        "best_mae": float(min(weight_mae)),
        "mae_improvement": float(weight_mae[0] - weight_mae[-1]),
        "mae_improvement_percent": float((weight_mae[0] - weight_mae[-1]) / weight_mae[0] * 100 if weight_mae[0] > 0 else 0),
        "per_class_stats": {
            "best_class": min(final_class_mae.items(), key=lambda x: x[1])[0],
            "worst_class": max(final_class_mae.items(), key=lambda x: x[1])[0],
            "best_mae": float(min(class_mae_values)),
            "worst_mae": float(max(class_mae_values)),
            "mean_class_mae": float(np.mean(class_mae_values)),
            "std_class_mae": float(np.std(class_mae_values)),
            "per_class_mae": final_class_mae
        }
    }

def _get_class_distribution(train_dataloader, val_dataloader, label_to_idx):
    """
    Calculate the distribution of classes in the dataset
    
    Args:
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        label_to_idx: Mapping from label names to indices
        
    Returns:
        dict: Class distribution information
    """
    # Initialize counters
    train_counts = {label: 0 for label in label_to_idx.keys()}
    val_counts = {label: 0 for label in label_to_idx.keys()}
    
    # Convert index to label
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    # Count training samples
    for _, labels, _ in train_dataloader:
        for label_idx in labels.numpy():
            label = idx_to_label[label_idx.item()]
            train_counts[label] += 1
    
    # Count validation samples
    for _, labels, _ in val_dataloader:
        for label_idx in labels.numpy():
            label = idx_to_label[label_idx.item()]
            val_counts[label] += 1
    
    # Calculate total counts
    total_counts = {label: train_counts[label] + val_counts[label] for label in label_to_idx.keys()}
    
    # Calculate percentages
    total_samples = sum(total_counts.values())
    percentages = {label: count / total_samples * 100 for label, count in total_counts.items()}
    
    return {
        "train": train_counts,
        "validation": val_counts,
        "total": total_counts,
        "class_percentages": {k: float(v) for k, v in percentages.items()}
    }
    
# ===================== MAIN EXECUTION =====================

if __name__ == '__main__':
    try:
        print("Starting Kaggle food recognition and weight estimation training...")
        
        # Record the start time for later calculation of total training time
        start_time = time.time()
        
        # Define default paths for Kaggle
        DATA_PATH = "/kaggle/input/data-set-labeld-weights"
        MODEL_DIR = "/kaggle/working"
        
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description="Train food recognition and weight estimation model in Kaggle")
        parser.add_argument("--csv_path", type=str, default=f"{DATA_PATH}/labels.csv", 
                           help="Path to the CSV file with annotations")
        parser.add_argument("--images_dir", type=str, default=f"{DATA_PATH}/image_set_2", 
                           help="Path to the directory with images")
        parser.add_argument("--epochs", type=int, default=50, 
                           help="Number of training epochs")
        parser.add_argument("--batch_size", type=int, default=16, 
                           help="Batch size for training")
        parser.add_argument("--lr", type=float, default=5e-5, 
                           help="Learning rate")
        parser.add_argument("--model_dir", type=str, default=MODEL_DIR, 
                           help="Directory to save the model")
        parser.add_argument("--num_workers", type=int, default=0, 
                           help="Number of workers for data loading")
        parser.add_argument("--lr_strategy", type=str, default="one_cycle",
                            choices=["one_cycle", "cosine", "step"],
                            help="Learning rate strategy to use")
        parser.add_argument("--early_stopping", type=int, default=10,
                            help="Number of epochs to wait for improvement before early stopping (0 to disable)")
        parser.add_argument("--reduce_lr", type=int, default=5,
                            help="Number of epochs to wait for improvement before reducing learning rate (0 to disable)")
        
        # Fix for Jupyter/Kaggle/Colab environments: ignore Jupyter's internal arguments
        # This handles the case where Jupyter passes -f /.local/share/jupyter/runtime/kernel-*.json
        def is_jupyter_environment():
            # Check for -f argument pattern which is commonly passed by Jupyter
            for i, arg in enumerate(sys.argv):
                if arg == '-f' and i+1 < len(sys.argv) and '.json' in sys.argv[i+1]:
                    return True
            
            # Also check for common Jupyter environment variables
            if 'JUPYTER_CONFIG_DIR' in os.environ or 'COLAB_GPU' in os.environ:
                return True
                
            # Check for in-process Jupyter detection
            try:
                # This will only work if IPython is installed
                import IPython
                if IPython.get_ipython() is not None:
                    return True
            except (ImportError, NameError):
                pass
                
            return False
        
        if is_jupyter_environment():
            print("Jupyter/Kaggle/Colab environment detected: filtering out Jupyter-specific arguments")
            # Parse only known arguments, ignoring others (like -f)
            args, unknown = parser.parse_known_args()
            if unknown:
                print(f"Ignored arguments: {unknown}")
        else:
            args = parser.parse_args()
        
        print(f"Configuration: {args}")
        
        # Ensure model directory exists
        os.makedirs(args.model_dir, exist_ok=True)
        
        # Verify files exist
        if not os.path.exists(args.csv_path):
            raise FileNotFoundError(f"CSV file not found at {args.csv_path}")
        
        if not os.path.exists(args.images_dir):
            raise FileNotFoundError(f"Images directory not found at {args.images_dir}")
        
        # Prepare data
        train_dataloader, val_dataloader, label_to_idx, full_dataset = prepare_data(
            args.csv_path, 
            args.images_dir, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers
        )
        
        # Get device
        device = get_device()
        
        # Initialize model
        num_classes = len(label_to_idx)
        print(f"Initializing model for {num_classes} food classes")
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
            full_dataset=full_dataset,
            lr_strategy=args.lr_strategy,
            best_lr=args.lr,
            early_stopping_patience=args.early_stopping,
            reduce_lr_patience=args.reduce_lr
        )
        
        # Save training log as JSON (ensuring it's JSON serializable)
        json_compatible_logs = {}
        for key, values in training_logs.items():
            if key in ["per_class_precision", "per_class_recall", "per_class_f1", "per_class_mae"]:
                json_compatible_logs[key] = {k: [float(x) for x in v] for k, v in values.items()}
            elif key == "confusion_matrix":
                json_compatible_logs[key] = values  # Already converted to list in _validate_epoch
            else:
                json_compatible_logs[key] = [float(x) for x in values]
        
        log_path = os.path.join(args.model_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(json_compatible_logs, f, indent=4)
        print(f"Training log saved to {log_path}")
        
        # Save comprehensive results in a separate JSON file
        # Get the best metrics
        best_epoch_idx = np.argmax(training_logs["val_accuracy"])
        best_accuracy = training_logs["val_accuracy"][best_epoch_idx]
        corresponding_mae = training_logs["weight_mae"][best_epoch_idx]
        best_epoch = training_logs["epochs"][best_epoch_idx]
        
        min_mae_idx = np.argmin(training_logs["weight_mae"])
        best_mae = training_logs["weight_mae"][min_mae_idx]
        corresponding_accuracy = training_logs["val_accuracy"][min_mae_idx]
        best_mae_epoch = training_logs["epochs"][min_mae_idx]
        
        # Calculate best composite score
        composite_scores = [acc - (mae/100) for acc, mae in 
                           zip(training_logs["val_accuracy"], training_logs["weight_mae"])]
        best_composite_idx = np.argmax(composite_scores)
        best_composite_score = composite_scores[best_composite_idx]
        best_composite_accuracy = training_logs["val_accuracy"][best_composite_idx]
        best_composite_mae = training_logs["weight_mae"][best_composite_idx]
        best_composite_epoch = training_logs["epochs"][best_composite_idx]
        
        # Get per-class best metrics
        best_class_metrics = _get_best_class_metrics(training_logs)
        
        # Get final epochs per-class metrics
        final_per_class_metrics = {
            "precision": {k: v[-1] for k, v in training_logs["per_class_precision"].items()},
            "recall": {k: v[-1] for k, v in training_logs["per_class_recall"].items()},
            "f1": {k: v[-1] for k, v in training_logs["per_class_f1"].items()},
            "mae": {k: v[-1] for k, v in training_logs["per_class_mae"].items()}
        }
        
        # Get the final convergence information
        convergence_info = _analyze_convergence(training_logs)
        
        # Calculate statistics about weight estimation
        weight_stats = _calculate_weight_estimation_stats(training_logs)
        
        # System specs and training environment
        try:
            import platform
            import torch.cuda as cuda
            import psutil
            
            system_info = {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "cpu": platform.processor(),
                "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
                "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else 0,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else "None",
            }
        except Exception as sys_info_err:
            system_info = {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "gpu": f"Unknown (Error retrieving system specs: {str(sys_info_err)})"
            }
        
        # Compile results
        results = {
            "training_config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "lr_strategy": args.lr_strategy,
                "model": "EfficientNet-B0",
                "dataset_size": len(train_dataloader.dataset) + len(val_dataloader.dataset),
                "train_size": len(train_dataloader.dataset),
                "val_size": len(val_dataloader.dataset),
                "num_classes": len(label_to_idx),
                "training_time_minutes": (time.time() - start_time) / 60,
                "steps_per_epoch": len(train_dataloader)
            },
            "best_metrics": {
                "best_accuracy": {
                    "value": float(best_accuracy),
                    "epoch": int(best_epoch),
                    "corresponding_mae": float(corresponding_mae)
                },
                "best_mae": {
                    "value": float(best_mae),
                    "epoch": int(best_mae_epoch),
                    "corresponding_accuracy": float(corresponding_accuracy)
                },
                "best_composite_score": {
                    "value": float(best_composite_score),
                    "epoch": int(best_composite_epoch),
                    "accuracy": float(best_composite_accuracy),
                    "mae": float(best_composite_mae)
                },
                "per_class_best": best_class_metrics
            },
            "final_metrics": {
                "accuracy": float(training_logs["val_accuracy"][-1]),
                "mae": float(training_logs["weight_mae"][-1]),
                "loss": float(training_logs["val_loss"][-1]),
                "composite_score": float(composite_scores[-1]),
                "per_class": final_per_class_metrics
            },
            "convergence_analysis": convergence_info,
            "weight_estimation_stats": weight_stats,
            "class_information": {
                "num_classes": len(label_to_idx),
                "classes": list(label_to_idx.keys()),
                "class_distribution": _get_class_distribution(train_dataloader, val_dataloader, label_to_idx)
            },
            "system_info": system_info,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save the comprehensive results
        results_path = os.path.join(args.model_dir, "training_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Comprehensive results saved to {results_path}")
        
        print("Training completed successfully!")
        
    except Exception as e:
        import traceback
        print("ERROR: An exception occurred during training:")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        print("Traceback:")
        traceback.print_exc()

def load_best_model(model_path, num_classes, device='cpu'):
    """
    Load the best saved model for inference.
    
    Args:
        model_path (str): Path to the saved model
        num_classes (int): Number of food classes
        device (str): Device to load the model to ('cpu' or 'cuda')
        
    Returns:
        tuple: (model, label_to_idx) - The loaded model and label-to-index mapping
    """
    if not os.path.exists(model_path):
        print(f"No model found at {model_path}")
        return None, None
    
    try:
        # Create a new model instance
        model = MultiTaskNet(num_classes)
        
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Extract label mapping
        label_to_idx = checkpoint.get('label_to_idx', {})
        
        print(f"Loaded model from {model_path}")
        print(f"Model performance:")
        print(f"- Validation accuracy: {checkpoint.get('val_accuracy', 0):.4f}")
        print(f"- Weight MAE: {checkpoint.get('val_mae', 0):.4f}g")
        print(f"- Composite score: {checkpoint.get('composite_score', 0):.4f}")
        
        return model, label_to_idx
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
