from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.config import IMG_SIZE, BATCH_SIZE


def load_dataset(path, train=False):
    if train:
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])

    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=train)

    return dataset, loader
