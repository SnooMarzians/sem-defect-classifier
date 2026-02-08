# src/model.py

import torch.nn as nn
from torchvision import models
from src.config import NUM_CLASSES

def get_model():
    model = models.efficientnet_b0(pretrained=True)

    # Replace classifier head
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        NUM_CLASSES
    )

    return model
