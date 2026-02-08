import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.dataset import load_dataset
from src.model import get_model
from src.config import MODEL_PATH, EPOCHS, LEARNING_RATE, CLASS_NAMES


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)

    # Load datasets
    train_ds, train_loader = load_dataset("data/train", train=True)
    val_ds, val_loader = load_dataset("data/val", train=False)

    model = get_model().to(device)

    # 🔥 Class-weighted loss (IMPORTANT for defect detection)
    class_weights = torch.tensor([
        0.5,  # clean
        2.0,  # bridges
        2.0,  # gaps
        2.5,  # cracks
        1.5,  # foreign_materials
        1.8,  # cmp
        1.8,  # ler
        1.5   # other
    ]).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print("✅ Best model saved")

    print("Training complete")


if __name__ == "__main__":
    train()
