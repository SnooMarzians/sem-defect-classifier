import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from src.config import NUM_CLASSES


def get_model():
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        NUM_CLASSES
    )

    return model
