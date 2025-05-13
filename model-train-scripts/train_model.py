#!/usr/bin/env python3
"""
Training script for food recognition and weight estimation model.
"""

import pandas as pd
import os
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score

def parse_args():
    parser = argparse.ArgumentParser(description="Train food recognition and weight estimation model")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to the CSV file with annotations")
    parser.add_argument("--images_dir", type=str, default=None, help="Path to the directory with images")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model_dir", type=str, default=None, help="Directory to save the model")
    return parser.parse_args()

# Parse command line arguments
args = parse_args()

# Define paths
master_thesis_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Set default paths if not provided via command line
if args.csv_path is None:
    args.csv_path = os.path.join(master_thesis_dir, "csvfiles", "combined_dataset_labels_ready.csv")

if args.images_dir is None:
    args.images_dir = os.path.join(master_thesis_dir, "ordered_dataset_foods_ready")

if args.model_dir is None:
    args.model_dir = os.path.join(master_thesis_dir, "models")

print(f"CSV path: {args.csv_path}")
print(f"Images directory: {args.images_dir}")
print(f"Model save directory: {args.model_dir}")

# Create model save directory
os.makedirs(args.model_dir, exist_ok=True)

# Load CSV
try:
    df = pd.read_csv(args.csv_path)
    print(f"Successfully loaded {len(df)} records from {args.csv_path}")
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

# Create label-to-index mapping
label_to_idx = {label: idx for idx, label in enumerate(df['labels'].unique())}
df['label_idx'] = df['labels'].map(label_to_idx)

print(f"Number of classes: {len(label_to_idx)}")
print(df.head())


from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch
import os

class FoodDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['image_name'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        label_idx = row['label_idx']
        weight = row['weight']
        
        return image, torch.tensor(label_idx, dtype=torch.long), torch.tensor(weight, dtype=torch.float32)


from torch.utils.data import DataLoader

# Set number of workers for data loading
num_workers = os.cpu_count() // 2  # Use half of available CPU cores
print(f"Using {num_workers} workers for data loading")

# Create dataset
dataset = FoodDataset(df, args.images_dir)
print(f"Dataset size: {len(dataset)} images")

# Create dataloader with batch size from command line arguments
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

print(f"Dataset size: {len(dataset)}")



import torch.nn as nn
import torchvision.models as models

class MultiTaskNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # remove original classifier
        
        self.classifier = nn.Linear(num_features, num_classes)  # classification head
        self.regressor = nn.Linear(num_features, 1)             # regression head
    
    def forward(self, x):
        features = self.backbone(x)
        class_logits = self.classifier(features)
        weight_pred = self.regressor(features).squeeze(1)
        return class_logits, weight_pred

num_classes = len(label_to_idx)
model = MultiTaskNet(num_classes)




import torch.optim as optim

criterion_class = nn.CrossEntropyLoss()
criterion_weight = nn.MSELoss()

# Use learning rate from command line arguments
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)  # Added weight decay for regularization
print(f"Optimizer: Adam with learning rate {args.lr}, weight decay 1e-5")



from torch.utils.data import random_split
import os
from sklearn.metrics import accuracy_score, mean_absolute_error

# Create model save directory
model_save_dir = os.path.join(master_thesis_dir, "models")
os.makedirs(model_save_dir, exist_ok=True)

# Split data into train and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

print(f"Training on {train_size} samples, validating on {val_size} samples")
print(f"Batch size: {args.batch_size}")
print(f"Epochs: {args.epochs}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model.to(device)

num_epochs = args.epochs
best_val_loss = float('inf')

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
        
        total_loss = loss_class + loss_weight  # can add weight factors if needed
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
    
    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model_path = os.path.join(model_save_dir, "best_model.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss,
            'val_accuracy': val_accuracy,
            'val_mae': val_mae,
            'label_to_idx': label_to_idx
        }, model_path)
        print(f"Model saved to {model_path}")

print("Training completed!")
