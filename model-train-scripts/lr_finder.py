#!/usr/bin/env python3
"""
Learning Rate Finder for food recognition and weight estimation model.
This script helps find an optimal learning rate by training the model with
exponentially increasing learning rates and observing the loss behavior.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.multiprocessing as mp

from data import prepare_data
from model import MultiTaskNet
from utils import parse_args, setup_paths, get_device

class LRFinder:
    """
    Learning rate finder class that implements the technique described in the
    paper "Cyclical Learning Rates for Training Neural Networks" by Leslie N. Smith.
    """
    def __init__(self, model, train_dataloader, criterion_class, criterion_weight,
                 device, start_lr=1e-7, end_lr=10, num_iterations=100, weight_class=0.7):
        self.model = model
        self.train_dataloader = train_dataloader
        self.criterion_class = criterion_class
        self.criterion_weight = criterion_weight
        self.device = device
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_iterations = min(num_iterations, len(train_dataloader))
        self.weight_class = weight_class
        self.weight_weight = 1.0 - weight_class
        
        # Calculate the multiplication factor for learning rate increase
        self.lr_factor = (end_lr / start_lr) ** (1 / (num_iterations - 1))
        
        # Prepare for storing results
        self.learning_rates = []
        self.losses = []
        self.best_lr = None
        
    def find(self):
        """Run the learning rate finder experiment."""
        # Initialize model in training mode
        self.model.train()
        
        # Set up optimizer with initial learning rate
        optimizer = optim.Adam(self.model.parameters(), lr=self.start_lr)
        
        # Get a fresh iterator for the data
        data_iter = iter(self.train_dataloader)
        
        # Log for progress
        print(f"Starting LR finder from {self.start_lr} to {self.end_lr} over {self.num_iterations} iterations")
        pbar = tqdm(range(self.num_iterations), desc="Finding optimal learning rate")
        
        # Keep track of the smoothed loss for finding the optimal learning rate
        smoothed_loss = None
        
        for iteration in pbar:
            # Get a batch of data
            try:
                images, labels, weights = next(data_iter)
            except StopIteration:
                # Restart iteration if we run out of data
                data_iter = iter(self.train_dataloader)
                images, labels, weights = next(data_iter)
            
            # Move data to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            weights = weights.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs_class, outputs_weight = self.model(images)
            
            # Calculate losses
            loss_class = self.criterion_class(outputs_class, labels)
            loss_weight = self.criterion_weight(outputs_weight, weights)
            
            # Combined loss with weights
            loss = self.weight_class * loss_class + self.weight_weight * loss_weight
            
            # Backward pass
            loss.backward()
            
            # Apply gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Take a step
            optimizer.step()
            
            # Record the learning rate and loss
            current_lr = optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Apply smoothing to the loss
            if smoothed_loss is None:
                smoothed_loss = loss.item()
            else:
                smoothed_loss = 0.98 * smoothed_loss + 0.02 * loss.item()
            
            self.losses.append(smoothed_loss)
            
            # Update the progress bar
            pbar.set_postfix({"lr": current_lr, "loss": smoothed_loss})
            
            # Increase the learning rate for the next iteration
            for param_group in optimizer.param_groups:
                param_group['lr'] *= self.lr_factor
        
        # Find the point of steepest decrease in the loss
        self._find_best_lr()
        
        return self.learning_rates, self.losses
    
    def _find_best_lr(self):
        """Find the learning rate with the steepest negative gradient in the loss."""
        # Convert to numpy arrays for easier manipulation
        lrs = np.array(self.learning_rates)
        losses = np.array(self.losses)
        
        # Calculate the gradient of the loss curve
        gradients = np.gradient(losses) / np.gradient(lrs)
        
        # Find the point with the steepest negative gradient
        # We'll ignore the first few and last few points for stability
        start_idx = len(gradients) // 10  # Skip first 10%
        end_idx = int(len(gradients) * 0.9)  # Skip last 10%
        
        section = gradients[start_idx:end_idx]
        if len(section) == 0:
            # Fallback in case of very short runs
            self.best_lr = self.learning_rates[len(self.learning_rates) // 2]
        else:
            idx = start_idx + np.argmin(section)
            self.best_lr = self.learning_rates[idx]
        
        # Recommend a learning rate slightly lower than where the minimum occurs
        self.best_lr = self.best_lr / 10.0
        
        print(f"Minimum gradient found at lr={self.learning_rates[idx]}")
        print(f"Recommended learning rate: {self.best_lr}")
        
        return self.best_lr
    
    def plot(self, save_path=None):
        """Plot the learning rate finder results."""
        plt.figure(figsize=(10, 6))
        plt.semilogx(self.learning_rates, self.losses)
        plt.xlabel("Learning Rate (log scale)")
        plt.ylabel("Loss")
        plt.title("Learning Rate Finder Results")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        
        # Mark the recommended learning rate
        if self.best_lr is not None:
            plt.axvline(x=self.best_lr, color='r', linestyle='--')
            plt.text(self.best_lr, min(self.losses), f"Recommended LR: {self.best_lr:.1e}", 
                     rotation=90, verticalalignment='bottom')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
            
        plt.show()

def run_lr_finder():
    """Run the learning rate finder as a standalone tool."""
    # Required for multiprocessing on macOS
    mp.set_start_method('spawn', force=True)
    
    # Parse command-line arguments
    args = parse_args()
    args = setup_paths(args)
    
    # Prepare data
    train_dataloader, _, label_to_idx = prepare_data(
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
    
    # Loss functions
    criterion_class = nn.CrossEntropyLoss()
    criterion_weight = nn.MSELoss()
    
    # Create the learning rate finder
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
    
    # Run the finder
    lr_finder.find()
    
    # Create the plot directory if it doesn't exist
    plot_dir = os.path.join(args.model_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot and save the results
    plot_path = os.path.join(plot_dir, "lr_finder_results.png")
    lr_finder.plot(save_path=plot_path)
    
    # Save the recommended learning rate
    with open(os.path.join(args.model_dir, 'recommended_lr.txt'), 'w') as f:
        f.write(f"Recommended learning rate: {lr_finder.best_lr}")
    
    print(f"Learning rate finder complete. Recommended LR: {lr_finder.best_lr}")
    
    return lr_finder.best_lr

if __name__ == '__main__':
    run_lr_finder()
