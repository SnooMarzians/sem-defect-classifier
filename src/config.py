# src/config.py

# ===== DATA =====
TRAIN_DIR = "data/train"
VAL_DIR   = "data/val"
TEST_DIR  = "data/test"

# ===== IMAGE =====
IMG_SIZE = 224
IMAGE_SIZE = 224   # for infer.py compatibility

# ===== TRAINING =====
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4

# ===== CLASSES =====
CLASS_NAMES = [
    "bridges",
    "clean",
    "cmp",
    "cracks",
    "foreign_materials",
    "gaps",
    "ler",
    "other"
]

NUM_CLASSES = len(CLASS_NAMES)

# ===== MODEL =====
MODEL_PATH = "saved_models/efficientnet_b0.pth"
ONNX_PATH  = "saved_models/efficientnet_b0.onnx"
