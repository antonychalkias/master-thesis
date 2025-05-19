#!/usr/bin/env python3
"""
Model architecture for food recognition and weight estimation.
"""

import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b0, EfficientNet_B0_Weights

class MultiTaskNet(nn.Module):
    """
    Multi-task neural network with two heads:
    - Classification head for food recognition
    - Regression head for weight estimation
    
    Uses either ResNet50 or EfficientNet-B0 as the backbone with pretrained weights.
    """
    def __init__(self, num_classes, model_type='efficientnet_b0'):
        super().__init__()
        
        if model_type == 'resnet50':
            # Load pretrained ResNet50 as backbone
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove the final fully connected layer
            
            # Task-specific heads
            self.classifier = nn.Linear(num_features, num_classes)  # classification head
            self.weight_regressor = nn.Sequential(
                nn.Linear(num_features, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        else:
            # Default to EfficientNet-B0
            self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()  # Remove the final classifier
            
            # Task-specific heads
            self.classifier = nn.Linear(num_features, num_classes)  # classification head
            self.weight_regressor = nn.Sequential(
                nn.Linear(num_features, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape [batch_size, 3, height, width]
            
        Returns:
            class_logits: Classification logits of shape [batch_size, num_classes]
            weight_pred: Weight predictions of shape [batch_size]
        """
        features = self.backbone(x)
        class_logits = self.classifier(features)
        weight_pred = self.weight_regressor(features).squeeze(1)
        return class_logits, weight_pred
