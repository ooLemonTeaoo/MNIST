import torch


def flatten_for_mlp(images):
    # Converts (B, 1, 28, 28) → (B, 784)
    return images.view(images.size(0), -1)

