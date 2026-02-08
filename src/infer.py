import torch
import torchvision.transforms as transforms
import cv2
from src.model import get_model
from src.config import MODEL_PATH, CLASS_NAMES, IMAGE_SIZE

CONFIDENCE_THRESHOLD = 0.75

def infer_image(image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_model(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    img = cv2.imread(image_path)
    if img is None:
        print("Invalid image path")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    confidence = conf.item()
    predicted_class = CLASS_NAMES[pred.item()]

    if confidence < CONFIDENCE_THRESHOLD:
        print(f"Predicted OTHER (low confidence: {confidence:.2f})")
        return "other"

    print(f"Predicted {predicted_class} ({confidence:.2f})")
    return predicted_class
