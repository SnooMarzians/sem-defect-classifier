# src/config.py

MODEL_PATH = "saved_models/efficientnet_b0.pth"

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

NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = 224
