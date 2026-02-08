# src/config.py

from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Model
MODEL_PATH = BASE_DIR / "saved_models" / "efficientnet_b0.pth"

# Classes (ORDER MUST MATCH TRAINING)
CLASS_NAMES = [
    "clean",
    "bridges",
    "gaps",
    "cracks",
    "foreign_materials",
    "cmp",
    "ler",
    "other"
]

# Image settings
IMAGE_SIZE = 224
