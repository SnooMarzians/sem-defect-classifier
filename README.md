# SEM Defect Classification (Edge AI)

## Overview
This project implements an automated **semiconductor manufacturing defect classification system** using **SEM images** and **EfficientNet-B0**. The solution is designed for **high accuracy, low latency**, and **edge-level deployment**.

The system classifies SEM images into multiple defect categories and includes robust handling of unknown or uncommon defects via a confidence-based rejection mechanism.

---

## Defect Classes
- Clean
- Bridges
- Gaps (Opens)
- Cracks
- Foreign Materials
- CMP
- LER (Line Edge Roughness)
- Other (unknown / miscellaneous defects)

---

## Model Architecture
- **Backbone:** EfficientNet-B0 (ImageNet pretrained)
- **Input size:** 224 × 224
- **Framework:** PyTorch
- **Export format:** ONNX

EfficientNet-B0 was chosen for its optimal balance between accuracy and computational efficiency, making it suitable for edge deployment.

---

## Dataset Structure
The dataset is organized as follows:

data/
├── train/
├── val/
└── test/
├── clean
├── bridges
├── gaps
├── cracks
├── foreign_materials
├── cmp
├── ler
└── other


> **Note:** Dataset images are not included in this repository as per submission guidelines.

---

## Training
Training is performed using transfer learning with GPU acceleration.

```bash
python -m src.train

-> Evaluation

Model performance is evaluated using Accuracy, Precision, Recall, F1-score, and Confusion Matrix

python -m src.evaluate

-> Inference

Run inference on a single SEM image:

from src.infer import infer_image
infer_image("path/to/image.jpg")

Low-confidence predictions are automatically classified as other.

-> Edge Deployment

The trained model can be exported to ONNX format for deployment on edge platforms:

python -m src.export_onnx

-> Hardware & Platform

GPU: NVIDIA RTX 4060

Acceleration: CUDA-enabled PyTorch

Training Platform: Local workstation (no cloud used)

-> Future Improvements

Advanced open-set recognition for unknown defects

Model quantization for ultra-low-power edge devices

Multi-scale input support

-> License

This project is developed for academic and hackathon purposes.

---

