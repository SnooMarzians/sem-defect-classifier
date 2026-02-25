import os
import torch
import cv2
from torchvision import transforms

from src.model import get_model
from src.config import MODEL_PATH, CLASS_NAMES, IMG_SIZE


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === PATHS ===
    prediction_folder = "data/Hackathon_phase3_prediction_dataset"
    output_file = "prediction_results.txt"

    # === LOAD MODEL ===
    model = get_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # === SAME PREPROCESSING AS TRAINING ===
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    with torch.no_grad(), open(output_file, "w") as fout:
        for image_name in sorted(os.listdir(prediction_folder)):

            if not image_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            image_path = os.path.join(prediction_folder, image_name)

            # Read image
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Preprocess
            img = transform(img)
            img = img.unsqueeze(0).to(device)

            # Inference
            outputs = model(img)
            _, pred = torch.max(outputs, 1)

            predicted_class = CLASS_NAMES[pred.item()]

            # Write to file
            fout.write(f"{image_name} : {predicted_class}\n")

    print(f"✅ Predictions saved to {output_file}")


if __name__ == "__main__":
    evaluate()