import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
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


def train_model(model, train_loader, val_loader, device, num_epochs, lr, model_save_path, train_df):
    y_train = train_df['label_idx'].values
    num_classes = model.classifier.out_features
    present_classes = np.unique(y_train)
    class_weights_np = compute_class_weight(class_weight='balanced', classes=present_classes, y=y_train)
    full_class_weights = np.ones(num_classes)
    full_class_weights[present_classes] = class_weights_np
    class_weights_tensor = torch.tensor(full_class_weights, dtype=torch.float32).to(device)

    criterion_class = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.05)
    criterion_weight = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    logs = {"epochs": [], "train_loss": [], "val_loss": [], "val_accuracy": [], "weight_mae": []}
    best_metrics = {'val_loss': float('inf'), 'val_accuracy': 0, 'val_mae': float('inf')}
    patience = 5
    epochs_no_improve = 0

    model_path = os.path.join(model_save_path, "best_model.pth")
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            best_metrics['val_loss'] = checkpoint.get('val_loss', float('inf'))
            best_metrics['val_accuracy'] = checkpoint.get('val_accuracy', 0)
            best_metrics['val_mae'] = checkpoint.get('val_mae', float('inf'))
            print("✅ Loaded previous best model checkpoint.")
        except Exception as e:
            print(f"Warning: Failed to load existing checkpoint: {e}")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels, weights in train_loader:
            images, labels, weights = images.to(device), labels.to(device), weights.to(device)
            optimizer.zero_grad()
            outputs_class, outputs_weight = model(images)
            loss = 0.7 * criterion_class(outputs_class, labels) + 0.3 * criterion_weight(outputs_weight, weights)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        all_preds, all_labels, all_weight_preds, all_weight_true = [], [], [], []
        with torch.no_grad():
            for images, labels, weights in val_loader:
                images, labels, weights = images.to(device), labels.to(device), weights.to(device)
                outputs_class, outputs_weight = model(images)
                loss = criterion_class(outputs_class, labels) + criterion_weight(outputs_weight, weights)
                val_loss += loss.item()
                _, preds = torch.max(outputs_class, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_weight_preds.extend(outputs_weight.cpu().numpy())
                all_weight_true.extend(weights.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        mae = mean_absolute_error(all_weight_true, all_weight_preds)
        logs["epochs"].append(epoch + 1)
        logs["train_loss"].append(train_loss / len(train_loader))
        avg_val_loss = val_loss / len(val_loader)
        logs["val_loss"].append(avg_val_loss)
        logs["val_accuracy"].append(acc)
        logs["weight_mae"].append(mae)

        improved = False
        if acc > best_metrics['val_accuracy']:
            improved = True
        elif acc >= best_metrics['val_accuracy'] * 0.95:
            if avg_val_loss < best_metrics['val_loss'] * 0.9 or mae < best_metrics['val_mae'] * 0.9:
                improved = True

        if improved:
            best_metrics.update({'val_loss': avg_val_loss, 'val_accuracy': acc, 'val_mae': mae})
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': avg_val_loss,
                'val_accuracy': acc,
                'val_mae': mae
            }, model_path)
            print(f"\n✅ Saved model at epoch {epoch + 1}: Val Loss={avg_val_loss:.4f}, Acc={acc:.4f}, MAE={mae:.2f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

        print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f} Val Loss={avg_val_loss:.4f} Acc={acc:.4f} MAE={mae:.2f}")
        scheduler.step()

    with open(os.path.join(model_save_path, "training_log.json"), "w") as f:
        json.dump({k: [float(x) if isinstance(x, (np.floating, np.float32)) else x for x in v] for k, v in logs.items()}, f, indent=4)


if __name__ == '__main__':
    DATA_PATH = "/kaggle/input/data-correctg"
    csv_path = f"{DATA_PATH}/csvfiles/latest.csv"
    images_dir = f"{DATA_PATH}/images"
    model_dir = "/kaggle/working"
    os.makedirs(model_dir, exist_ok=True)

    train_loader, val_loader, label_to_idx, train_df = prepare_data(
        csv_path, images_dir, batch_size=16, num_workers=4
    )

    device = torch.device("cuda")
    model = MultiTaskNet(num_classes=len(label_to_idx))

    model.to(device)

    train_model(model, train_loader, val_loader, device, num_epochs=20, lr=1e-4, model_save_path=model_dir, train_df=train_df)

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