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
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt


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
    best_val_loss = None
    best_acc = 0
    patience = 5
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels, weights in train_loader:
            images, labels, weights = images.to(device), labels.to(device), weights.to(device)
            optimizer.zero_grad()
            outputs_class, outputs_weight = model(images)
            loss = criterion_class(outputs_class, labels) + 0.01 * criterion_weight(outputs_weight, weights)
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
        logs["epochs"].append(int(epoch + 1))
        logs["train_loss"].append(float(train_loss / len(train_loader)))
        logs["val_loss"].append(float(val_loss / len(val_loader)))
        logs["val_accuracy"].append(float(acc))
        logs["weight_mae"].append(float(mae))

        current_val_loss = val_loss / len(val_loader)
        if best_val_loss is None or current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_model_path = os.path.join(model_save_path, "best_model.pth")
            if not os.path.exists(best_model_path) or current_val_loss < best_val_loss:
                torch.save(model.state_dict(), best_model_path)
                print(f"\n✅ Saved best model at epoch {epoch + 1} with Val Loss={current_val_loss:.4f}, Acc={acc:.4f}, MAE={mae:.2f}")

        if acc > best_acc:
            best_acc = acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

        print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f} Val Loss={current_val_loss:.4f} Acc={acc:.4f} MAE={mae:.2f}")
        scheduler.step()

    with open(os.path.join(model_save_path, "training_log.json"), "w") as f:
        json.dump(logs, f, indent=4)


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
    checkpoint_path = os.path.join(model_dir, "best_model.pth")

    if os.path.exists(checkpoint_path):
        print("Loading best model for continued training...")
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        print("Training from scratch.")

    model.to(device)

    train_model(model, train_loader, val_loader, device, num_epochs=20, lr=1e-4, model_save_path=model_dir, train_df=train_df)

    # --- Evaluation block ---
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

    # Classification Report
    print("\nClassification Report:\n")
    print(classification_report(
        all_labels,
        all_preds,
        labels=list(idx_to_label.keys()),
        target_names=list(idx_to_label.values()),
        zero_division=0
    ))




# Bach 1 results
# ✅ Saved best model at epoch 1 with Val Loss=76272.4651, Acc=0.1881, MAE=213.68
# Epoch 1: Train Loss=3709.2706 Val Loss=76272.4651 Acc=0.1881 MAE=213.68
# Epoch 2: Train Loss=3425.3198 Val Loss=76441.5947 Acc=0.1782 MAE=213.76
# Epoch 3: Train Loss=2569.4999 Val Loss=76495.2275 Acc=0.0000 MAE=212.86
# Epoch 4: Train Loss=1244.7807 Val Loss=75916.2290 Acc=0.0000 MAE=209.91
# Epoch 5: Train Loss=1616.6144 Val Loss=76845.8707 Acc=0.0495 MAE=214.42
# Early stopping triggered.

# Classification Report:

#                                               precision    recall  f1-score   support

#                                        bagel       0.00      0.00      0.00         0
#                                    croissant       0.00      0.00      0.00         0
#                             blueberry_muffin       0.00      0.00      0.00         0
#                                         corn       0.00      0.00      0.00         0
#                                     broccoli       0.00      0.00      0.00         0
#                                      avocado       0.00      0.00      0.00         0
#                               chicken_nugget       0.00      0.00      0.00         0
#                                      biscuit       0.00      0.00      0.00         0
#                                        apple       0.00      0.00      0.00         0
#                                   strawberry       0.00      0.00      0.00         0
#                                cinnamon_roll       0.79      0.58      0.67        19
#                        Cucumber,Chicken,Rice       0.00      0.00      0.00         1
#         Cucumber,Mushrooms,Salad Leaves,Rice       0.00      0.00      0.00         1
# Cucumber,Mushrooms,Chicken,Rice,Salad Leaves       0.00      0.00      0.00         1
#              Rice,Mushrooms,Cucumber,Chicken       0.00      0.00      0.00         1
#                                        Bread       0.00      0.00      0.00         4
#                                         Feta       0.00      0.00      0.00        13
#                    Rice,Salad Leaves,Chicken       0.00      0.00      0.00         1
#     Eggs,Mushrooms,Salad Leaves,Rice,Chicken       0.00      0.00      0.00         1
#                                         Eggs       0.00      0.00      0.00         3
#                                         Rice       0.00      0.00      0.00         5
#                  Potatos,Green Beans,Carrots       0.00      0.00      0.00         1
#              Rice,Mushrooms,Chicken,Cucumber       0.00      0.00      0.00         1
#                                      Burgers       0.00      0.00      0.00         3
#                  Peas,Potatos,Onions,Carrots       0.00      0.00      0.00         1
#                    Rice,Chicken,Salad Leaves       0.00      0.00      0.00         1
#                                 Peas,Potatos       0.00      0.00      0.00         2
#                                        Pasta       0.00      0.00      0.00        10
#                              Gemista,Carrots       0.00      0.00      0.00         3
#                             Giouvetsi,Cheese       0.00      0.00      0.00         1
#                                  Snails,Rice       0.00      0.00      0.00         1
#                                      Chicken       0.00      0.00      0.00         2
#                                 Rice,Carrots       0.00      0.00      0.00         1
#                                 Potatos,Lamb       0.00      0.00      0.00         1
#                        Rice,Chicken,Cucumber       0.00      0.00      0.00         1
#                         Peas,Potatos,Carrots       0.00      0.00      0.00         2
#                                 Okra,Potatos       0.00      0.00      0.00         3
#                                     Broccoli       0.00      0.00      0.00         2
#                                 Potatos,Peas       0.00      0.00      0.00         1
#                                  Green Beans       0.00      0.00      0.00         4
#                                  Cheese,Eggs       0.00      0.00      0.00         1
#                                      Gemista       0.00      0.00      0.00         3
#                                         Peas       0.00      0.00      0.00         1
#                                        Other       0.00      0.00      0.00         2
#                                      Carrots       0.00      0.00      0.00         1
#                              Potatos,Burgers       0.00      0.00      0.00         1
#                                    Giouvetsi       0.00      0.00      0.00         1

#                                    micro avg       0.11      0.11      0.11       101
#                                    macro avg       0.02      0.01      0.01       101
#                                 weighted avg       0.15      0.11      0.13       101
