import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F


# converts class labels (numbers) into vectors of 0 and 1
def one_hot(labels, num_classes=10):
    return F.one_hot(labels, num_classes=num_classes).float()

def flatten_for_mlp(images):
    # Converts (B, 1, 28, 28) → (B, 784)
    return images.view(images.size(0), -1)

def get_mnist_loaders(batch_size=64, val_split=0.1):
    # Transform image(0 - 255) to PyTorch tensor(0 - 1)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load dataset for training
    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    # Load dataset for testing
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    # Train/validation split
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size

    train_ds, val_ds = random_split(train_dataset, [train_size, val_size])

    # DataLoaders
    # Split into batch due to memory limitation
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader