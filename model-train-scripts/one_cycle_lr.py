#!/usr/bin/env python3
"""
Implementation of the One Cycle Learning Rate scheduler for
food recognition and weight estimation model.
"""

import math
import matplotlib.pyplot as plt
import numpy as np

class OneCycleLR:
    """
    One Cycle Learning Rate Scheduler as proposed by Leslie Smith
    in the paper "Super-Convergence: Very Fast Training of Neural Networks
    Using Large Learning Rates".
    
    This scheduler:
    1. Implements a learning rate schedule that first increases linearly from
       initial_lr to max_lr, then decreases back to a very small value
    2. Simultaneously decreases momentum from max_momentum to min_momentum,
       then increases it back
    """
    def __init__(self, optimizer, max_lr, total_epochs, steps_per_epoch,
                 pct_start=0.3, div_factor=25.0, final_div_factor=1e4,
                 min_momentum=0.85, max_momentum=0.95):
        """
        Initialize the scheduler.
        
        Args:
            optimizer: The optimizer for which to adjust the learning rate
            max_lr: Maximum learning rate
            total_epochs: Total number of epochs for training
            steps_per_epoch: Number of steps (batches) in an epoch
            pct_start: Percentage of the cycle spent increasing the learning rate
            div_factor: Determines the initial learning rate via initial_lr = max_lr/div_factor
            final_div_factor: Determines the minimum learning rate via min_lr = initial_lr/final_div_factor
            min_momentum: Minimum momentum value
            max_momentum: Maximum momentum value
        """
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_epochs * steps_per_epoch
        self.step_size_up = int(self.total_steps * pct_start)
        self.step_size_down = self.total_steps - self.step_size_up
        self.initial_lr = max_lr / div_factor
        self.min_lr = self.initial_lr / final_div_factor
        self.min_momentum = min_momentum
        self.max_momentum = max_momentum
        self.current_step = 0
        
        self.history = {
            'iterations': [],
            'learning_rate': [],
            'momentum': []
        }
        
        # Set initial learning rate and momentum
        self._set_lr(self.initial_lr)
        self._set_momentum(self.max_momentum)
        
    def _set_lr(self, lr):
        """Set the learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def _set_momentum(self, momentum):
        """Set the momentum for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            if 'betas' in param_group:
                # For Adam optimizer
                param_group['betas'] = (momentum, param_group['betas'][1])
            elif 'momentum' in param_group:
                # For SGD with momentum
                param_group['momentum'] = momentum
    
    def step(self):
        """Take a step in the scheduler."""
        if self.current_step >= self.total_steps:
            return
            
        self.current_step += 1
        
        # Calculate the current learning rate and momentum
        if self.current_step <= self.step_size_up:
            # Increasing learning rate phase
            lr_factor = self.current_step / self.step_size_up
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * lr_factor
            
            # Decreasing momentum phase
            momentum_factor = self.current_step / self.step_size_up
            momentum = self.max_momentum - (self.max_momentum - self.min_momentum) * momentum_factor
        else:
            # Decreasing learning rate phase
            down_step = self.current_step - self.step_size_up
            cosine_decay = 0.5 * (1 + math.cos(math.pi * down_step / self.step_size_down))
            lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
            
            # Increasing momentum phase
            momentum_factor = down_step / self.step_size_down
            momentum = self.min_momentum + (self.max_momentum - self.min_momentum) * momentum_factor
        
        # Update the learning rate and momentum
        self._set_lr(lr)
        self._set_momentum(momentum)
        
        # Record history
        self.history['iterations'].append(self.current_step)
        self.history['learning_rate'].append(lr)
        self.history['momentum'].append(momentum)
        
    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def get_momentum(self):
        """Get the current momentum."""
        if 'betas' in self.optimizer.param_groups[0]:
            return self.optimizer.param_groups[0]['betas'][0]
        elif 'momentum' in self.optimizer.param_groups[0]:
            return self.optimizer.param_groups[0]['momentum']
        return None
        
    def plot_schedule(self, save_path=None):
        """Plot the learning rate and momentum schedule."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot learning rate
        ax1.plot(self.history['iterations'], self.history['learning_rate'])
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Learning Rate')
        ax1.set_title('One Cycle LR Schedule')
        ax1.grid(True)
        
        # Plot momentum
        ax2.plot(self.history['iterations'], self.history['momentum'])
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Momentum')
        ax2.set_title('One Cycle Momentum Schedule')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Schedule plot saved to {save_path}")
            
        plt.show()
        
    def state_dict(self):
        """Return the state of the scheduler as a dictionary."""
        return {
            'current_step': self.current_step,
            'max_lr': self.max_lr,
            'initial_lr': self.initial_lr,
            'min_lr': self.min_lr,
            'step_size_up': self.step_size_up,
            'step_size_down': self.step_size_down,
            'min_momentum': self.min_momentum,
            'max_momentum': self.max_momentum,
            'history': self.history,
        }
        
    def load_state_dict(self, state_dict):
        """Load the scheduler state from a dictionary."""
        self.current_step = state_dict['current_step']
        self.max_lr = state_dict['max_lr']
        self.initial_lr = state_dict['initial_lr']
        self.min_lr = state_dict['min_lr']
        self.step_size_up = state_dict['step_size_up']
        self.step_size_down = state_dict['step_size_down']
        self.min_momentum = state_dict['min_momentum']
        self.max_momentum = state_dict['max_momentum']
        self.history = state_dict['history']
