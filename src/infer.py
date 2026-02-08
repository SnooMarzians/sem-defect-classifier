# src/infer.py

import torch
import cv2
import numpy as np
from torchvision import transforms

from src.model import get_model
from src.config import MODEL_PATH, CLASS_NAMES, IMAGE_SIZE

def infer_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = get_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print("Invalid image path")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    label = CLASS_NAMES[pred.item()]
    confidence = conf.item()

    print(f"Predicted {label} ({confidence:.2f})")
    return label
