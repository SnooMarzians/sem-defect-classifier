import torch
from src.model import get_model
from src.config import MODEL_PATH, ONNX_PATH, IMG_SIZE


def export_onnx():
    device = torch.device("cpu")

    model = get_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        opset_version=12
    )

    print("ONNX model exported:", ONNX_PATH)


if __name__ == "__main__":
    export_onnx()
