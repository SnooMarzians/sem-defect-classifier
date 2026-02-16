import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import load_dataset
from src.model import get_model
from src.config import MODEL_PATH, CLASS_NAMES


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_loader = load_dataset("data/test", train=False)

    model = get_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    print("Confusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    evaluate()
