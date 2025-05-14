# CUDA available: True
# Using device: cuda
# Epoch 1/20: Train Loss = 4415.5357, Val Loss = 4954.4272, Val Acc = 0.0000, Weight MAE = 53.34g
# Epoch 2/20: Train Loss = 5234.1350, Val Loss = 4834.2596, Val Acc = 0.0000, Weight MAE = 53.90g
# Epoch 3/20: Train Loss = 4317.4858, Val Loss = 4502.6738, Val Acc = 0.0476, Weight MAE = 52.72g
# Model saved to /kaggle/working/best_model.pth (Val Loss: 4502.6738, Val Acc: 0.0476, Val MAE: 52.72g)
# Epoch 4/20: Train Loss = 2737.1670, Val Loss = 4065.8627, Val Acc = 0.0000, Weight MAE = 49.04g
# Epoch 5/20: Train Loss = 2143.6895, Val Loss = 3999.0575, Val Acc = 0.0000, Weight MAE = 48.64g
# Epoch 6/20: Train Loss = 1839.5466, Val Loss = 4394.9641, Val Acc = 0.0476, Weight MAE = 52.09g
# Epoch 7/20: Train Loss = 1665.1224, Val Loss = 4268.4416, Val Acc = 0.0476, Weight MAE = 51.22g
# Epoch 8/20: Train Loss = 1198.9331, Val Loss = 3409.8324, Val Acc = 0.0476, Weight MAE = 44.54g
# Epoch 9/20: Train Loss = 803.7823, Val Loss = 3610.8154, Val Acc = 0.0476, Weight MAE = 45.62g
# Epoch 10/20: Train Loss = 967.0954, Val Loss = 3985.5651, Val Acc = 0.0476, Weight MAE = 48.49g
# Epoch 11/20: Train Loss = 747.9750, Val Loss = 2963.3466, Val Acc = 0.0000, Weight MAE = 41.61g
# Epoch 12/20: Train Loss = 544.9141, Val Loss = 3001.9075, Val Acc = 0.0476, Weight MAE = 39.07g
# Epoch 13/20: Train Loss = 631.6217, Val Loss = 3264.1064, Val Acc = 0.0000, Weight MAE = 41.15g
# Epoch 14/20: Train Loss = 513.0885, Val Loss = 3347.6038, Val Acc = 0.0000, Weight MAE = 42.40g
# Epoch 15/20: Train Loss = 466.7584, Val Loss = 1329.6108, Val Acc = 0.0238, Weight MAE = 26.40g
# Epoch 16/20: Train Loss = 496.3193, Val Loss = 2810.5298, Val Acc = 0.0000, Weight MAE = 36.71g
# Epoch 17/20: Train Loss = 472.0945, Val Loss = 2419.1306, Val Acc = 0.0000, Weight MAE = 32.69g
# Epoch 18/20: Train Loss = 328.6192, Val Loss = 1480.3718, Val Acc = 0.0000, Weight MAE = 26.59g
# Epoch 19/20: Train Loss = 296.8098, Val Loss = 939.0473, Val Acc = 0.0000, Weight MAE = 21.27g
# Epoch 20/20: Train Loss = 259.2639, Val Loss = 1323.9699, Val Acc = 0.0238, Weight MAE = 25.20g


import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


class MultiTaskNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(num_features, num_classes)
        self.regressor = nn.Linear(num_features, 1)

    def forward(self, x):
        features = self.backbone(x)
        class_logits = self.classifier(features)
        weight_pred = self.regressor(features).squeeze(1)
        return class_logits, weight_pred


class FoodDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['image_name'])
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='black')
        image = self.transform(image)
        label = torch.tensor(row['label_idx'], dtype=torch.long)
        weight = torch.tensor(row['weight'], dtype=torch.float32)
        return image, label, weight


def prepare_data(csv_path, images_dir, batch_size=16, num_workers=0):
    df = pd.read_csv(csv_path, sep=';', quotechar='"')
    label_to_idx = {label: idx for idx, label in enumerate(df['labels'].unique())}
    df['label_idx'] = df['labels'].map(label_to_idx)
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:].reset_index(drop=True)

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = FoodDataset(train_df, images_dir, transform=train_transform)
    val_dataset = FoodDataset(val_df, images_dir, transform=val_transform)

    class_counts = train_df['label_idx'].value_counts().reindex(range(len(label_to_idx)), fill_value=1).values
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = torch.tensor([weights[label] for label in train_df['label_idx']], dtype=torch.float)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, label_to_idx, train_df


def train_model(model, train_dataloader, val_dataloader, device, num_epochs, model_save_dir):
    criterion_class = nn.CrossEntropyLoss()
    criterion_weight = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

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

    model_path = os.path.join(model_save_dir, "best_model.pth")

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for images, labels, weights in train_dataloader:
            images, labels, weights = images.to(device), labels.to(device), weights.to(device)
            optimizer.zero_grad()
            outputs_class, outputs_weight = model(images)
            loss_class = criterion_class(outputs_class, labels)
            loss_weight = criterion_weight(outputs_weight, weights)
            total_loss = 0.7 * loss_class + 0.3 * loss_weight
            total_loss.backward()
            optimizer.step()
            running_train_loss += total_loss.item()

        avg_train_loss = running_train_loss / len(train_dataloader)

        model.eval()
        running_val_loss = 0.0
        all_preds, all_labels, all_weight_preds, all_weight_true = [], [], [], []
        with torch.no_grad():
            for images, labels, weights in val_dataloader:
                images, labels, weights = images.to(device), labels.to(device), weights.to(device)
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

        training_logs["epochs"].append(epoch + 1)
        training_logs["train_loss"].append(avg_train_loss)
        training_logs["val_loss"].append(avg_val_loss)
        training_logs["val_accuracy"].append(val_accuracy)
        training_logs["weight_mae"].append(val_mae)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'val_mae': val_mae
            }, model_path)
            print(f"Model saved to {model_path} (Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val MAE: {val_mae:.2f}g)")

        scheduler.step()

    return training_logs


if __name__ == '__main__':
    print("CUDA available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    DATA_PATH = "/kaggle/input/thesis-fine-tuned"
    csv_path = f"{DATA_PATH}/latest.csv"
    images_dir = f"{DATA_PATH}/images"
    model_dir = "/kaggle/working"
    os.makedirs(model_dir, exist_ok=True)

    train_loader, val_loader, label_to_idx, train_df = prepare_data(
        csv_path, images_dir, batch_size=16, num_workers=4
    )

    model = MultiTaskNet(num_classes=len(label_to_idx))
    model.to(device)

    train_model(model, train_loader, val_loader, device, num_epochs=20, model_save_dir=model_dir)

    model.eval()
    all_preds, all_labels, all_weight_preds, all_weight_true = [], [], [], []
    with torch.no_grad():
        for images, labels, weights in val_loader:
            images = images.to(device)
            outputs_class, outputs_weight = model(images)
            _, preds = torch.max(outputs_class, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_weight_preds.extend(outputs_weight.cpu().numpy())
            all_weight_true.extend(weights.numpy())

    idx_to_label = {v: k for k, v in label_to_idx.items()}
    print("\nClassification Report:\n")
    print(classification_report(
        all_labels,
        all_preds,
        labels=list(idx_to_label.keys()),
        target_names=list(idx_to_label.values()),
        zero_division=0
    ))
