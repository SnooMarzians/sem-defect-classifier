import os
import logging
import torch
from datetime import datetime

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.dataset import load_dataset
from src.model import get_model
from src.config import MODEL_PATH, CLASS_NAMES


# =========================
# LOGGING SETUP
# =========================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "hackathon_test_dataset_prediction.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # LOAD MODEL
    # -------------------------
    logger.info("Loading model...")
    model = get_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    logger.info("Model loaded successfully.\n")

    # -------------------------
    # LOAD DATASET
    # -------------------------
    logger.info("Loading hackathon test dataset...")
    test_dataset, test_loader = load_dataset("data/test", train=False)
    logger.info("Test dataset loaded.")
    logger.info(f"Total images found: {len(test_dataset)}\n")

    y_true = []
    y_pred = []

    # -------------------------
    # INFERENCE (PER IMAGE LOG)
    # -------------------------
    logger.info("Starting inference...\n")

    image_counter = 0
    total_images = len(test_dataset)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for i in range(len(images)):
                image_counter += 1

                filename = os.path.basename(test_dataset.samples[image_counter - 1][0])
                pred_class = CLASS_NAMES[preds[i].item()]

                logger.info(f"Image {image_counter}/{total_images}")
                logger.info(f"File: {filename}")
                logger.info(f"Predicted: {pred_class}\n")

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    logger.info("Inference complete.\n")

    # -------------------------
    # METRICS
    # -------------------------
    logger.info("Generating classification report...\n")

    accuracy = accuracy_score(y_true, y_pred)
    logger.info(f"Accuracy: {accuracy:.2f}\n")

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=2
    )
    logger.info(report)

    logger.info("Confusion Matrix:\n")
    cm = confusion_matrix(y_true, y_pred)
    logger.info(cm)


if __name__ == "__main__":
    main()
