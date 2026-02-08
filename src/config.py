# ===== Model & Training Configuration =====

NUM_CLASSES = 8

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

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 1e-4

# ===== Paths =====
MODEL_PATH = "saved_models/efficientnet_b0.pth"
ONNX_PATH = "saved_models/efficientnet_b0.onnx"
