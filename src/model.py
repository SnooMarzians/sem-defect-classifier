# src/model.py

import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from src.config import CLASS_NAMES


def get_model():
    """
    Returns EfficientNet-B0 model adapted for SEM defect classification
    """

    num_classes = len(CLASS_NAMES)

    # Load pretrained EfficientNet-B0
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model
