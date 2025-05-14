#!/usr/bin/env python3
"""
Script to compare different learning rate strategies for the food recognition model.
"""

import os
import subprocess
import json
import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np

def run_training(csv_path, images_dir, model_dir, lr_strategy, lr=None, epochs=20, batch_size=16):
    """Run training with specific learning rate strategy."""
    # Create the command
    cmd = [
        "python", "train.py",
        "--csv_path", csv_path,
        "--images_dir", images_dir,
        "--model_dir", model_dir,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr_strategy", lr_strategy
    ]
    
    # Add learning rate if specified
    if lr is not None:
        cmd.extend(["--lr", str(lr)])
    
    # Execute the training
    print(f"Starting training with {lr_strategy} strategy...")
    process = subprocess.run(cmd, check=True)
    
    # Return the path to the training log
    log_path = os.path.join(model_dir, "training_log.json")
    return log_path

def create_comparative_plots(log_paths, strategies, output_dir):
    """Create comparative plots for different learning rate strategies."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all logs
    logs = {}
    for path, strategy in zip(log_paths, strategies):
        try:
            with open(path, 'r') as f:
                logs[strategy] = json.load(f)
        except Exception as e:
            print(f"Error loading log {path}: {e}")
    
    if not logs:
        print("No valid logs found to compare.")
        return
    
    # Colors for different strategies
    colors = {
        'one_cycle': 'blue',
        'cosine': 'green',
        'step': 'red',
        'find': 'purple'
    }
    
    # Create comparative plots
    metrics = ["val_accuracy", "weight_mae", "val_loss"]
    titles = ["Validation Accuracy", "Weight MAE", "Validation Loss"]
    ylabels = ["Accuracy", "MAE (grams)", "Loss"]
    
    # Create a figure with subplots for each metric
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
        ax = axes[i]
        for strategy, log in logs.items():
            color = colors.get(strategy, 'black')
            ax.plot(log["epochs"], log[metric], color=color, marker='o', markersize=4, label=strategy)
        
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()
        
        # Use log scale for loss
        if metric == "val_loss":
            ax.set_yscale('log')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "lr_strategies_comparison.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Comparison plot saved to {plot_path}")
    
    # Create a learning rate comparison plot
    if all("learning_rates" in log for log in logs.values()):
        plt.figure(figsize=(10, 6))
        for strategy, log in logs.items():
            color = colors.get(strategy, 'black')
            plt.plot(log["epochs"], log["learning_rates"], color=color, marker='o', markersize=4, label=strategy)
        
        plt.title("Learning Rate Schedules")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        
        lr_path = os.path.join(output_dir, "learning_rate_comparison.png")
        plt.savefig(lr_path, dpi=300)
        print(f"Learning rate comparison plot saved to {lr_path}")
    
    # Create a summary table of best metrics
    summary = {}
    for strategy, log in logs.items():
        best_acc_idx = np.argmax(log["val_accuracy"])
        best_mae_idx = np.argmin(log["weight_mae"])
        best_loss_idx = np.argmin(log["val_loss"])
        
        summary[strategy] = {
            "best_accuracy": {
                "value": log["val_accuracy"][best_acc_idx],
                "epoch": log["epochs"][best_acc_idx]
            },
            "best_mae": {
                "value": log["weight_mae"][best_mae_idx],
                "epoch": log["epochs"][best_mae_idx]
            },
            "best_loss": {
                "value": log["val_loss"][best_loss_idx],
                "epoch": log["epochs"][best_loss_idx]
            }
        }
    
    # Save summary as JSON
    summary_path = os.path.join(output_dir, "strategies_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Summary saved to {summary_path}")
    
    # Print summary to console
    print("\nSummary of Results:")
    print("-" * 80)
    print(f"{'Strategy':<12} {'Best Accuracy':<20} {'Best MAE':<20} {'Best Loss':<20}")
    print("-" * 80)
    for strategy, metrics in summary.items():
        acc = f"{metrics['best_accuracy']['value']:.4f} (ep {metrics['best_accuracy']['epoch']})"
        mae = f"{metrics['best_mae']['value']:.2f}g (ep {metrics['best_mae']['epoch']})"
        loss = f"{metrics['best_loss']['value']:.2f} (ep {metrics['best_loss']['epoch']})"
        print(f"{strategy:<12} {acc:<20} {mae:<20} {loss:<20}")
    
    return summary

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare learning rate strategies")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file with annotations")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to the directory with images")
    parser.add_argument("--base_dir", type=str, default="models/lr_experiments", help="Base directory for experiment outputs")
    parser.add_argument("--strategies", type=str, nargs="+", default=["one_cycle", "cosine", "step"], 
                        help="Learning rate strategies to compare")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs per training run")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-5, help="Base learning rate")
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Create the experiment directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.base_dir, f"exp_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save the experiment parameters
    with open(os.path.join(exp_dir, "experiment_config.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Run training for each strategy
    log_paths = []
    strategies = []
    
    for strategy in args.strategies:
        # Create directory for this strategy
        strategy_dir = os.path.join(exp_dir, strategy)
        os.makedirs(strategy_dir, exist_ok=True)
        
        # Run training
        log_path = run_training(
            csv_path=args.csv_path,
            images_dir=args.images_dir,
            model_dir=strategy_dir,
            lr_strategy=strategy,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        log_paths.append(log_path)
        strategies.append(strategy)
        
        # Create individual visualizations
        subprocess.run([
            "python", "visualize_training_advanced.py",
            "--log_path", log_path,
            "--output_dir", os.path.join(strategy_dir, "plots")
        ])
    
    # Create comparative plots
    create_comparative_plots(log_paths, strategies, os.path.join(exp_dir, "comparisons"))
    
    print(f"\nExperiment completed successfully. Results saved to {exp_dir}")

if __name__ == "__main__":
    main()
