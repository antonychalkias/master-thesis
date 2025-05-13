import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error
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
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

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
    dataset = FoodDataset(df, images_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, label_to_idx, df


def train_model(model, train_loader, val_loader, device, num_epochs, lr, model_save_path, df):
    y_train = df['label_idx'].values
    class_weights_np = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.tensor(class_weights_np, dtype=torch.float32).to(device)

    criterion_class = nn.CrossEntropyLoss(weight=class_weights_tensor)
    criterion_weight = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    logs = {"epochs": [], "train_loss": [], "val_loss": [], "val_accuracy": [], "weight_mae": []}
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels, weights in train_loader:
            images, labels, weights = images.to(device), labels.to(device), weights.to(device)
            optimizer.zero_grad()
            outputs_class, outputs_weight = model(images)
            loss = 0.9 * criterion_class(outputs_class, labels) + 0.1 * criterion_weight(outputs_weight, weights)
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
        logs["val_loss"].append(val_loss / len(val_loader))
        logs["val_accuracy"].append(acc)
        logs["weight_mae"].append(mae)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_save_path, "best_model.pth"))

        print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f} Val Loss={val_loss:.4f} Acc={acc:.4f} MAE={mae:.2f}")
        scheduler.step()

    json_compatible_logs = {
        "epochs": [int(x) for x in logs["epochs"]],
        "train_loss": [float(x) for x in logs["train_loss"]],
        "val_loss": [float(x) for x in logs["val_loss"]],
        "val_accuracy": [float(x) for x in logs["val_accuracy"]],
        "weight_mae": [float(x) for x in logs["weight_mae"]],
    }

    with open(os.path.join(model_save_path, "training_log.json"), "w") as f:
        json.dump(json_compatible_logs, f, indent=4)


if __name__ == '__main__':
    DATA_PATH = "/kaggle/input/thesis-data"
    csv_path = f"{DATA_PATH}/ccsvfiles/latest_lab.csv"
    images_dir = f"{DATA_PATH}/images"
    model_dir = "/kaggle/working"
    os.makedirs(model_dir, exist_ok=True)

    train_loader, val_loader, label_to_idx, df = prepare_data(
        csv_path, images_dir, batch_size=16, num_workers=4
    )

    device = torch.device("cuda")

    model = MultiTaskNet(num_classes=len(label_to_idx))
    checkpoint_path = os.path.join(model_dir, "best_model.pth")
    
    if os.path.exists(checkpoint_path):
        print("Loading best model for continued training...")
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        print("Training from scratch.")

    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     model = nn.DataParallel(model)

    model.to(device)

    train_model(model, train_loader, val_loader, device, num_epochs=20, lr=1e-4, model_save_path=model_dir, df=df)