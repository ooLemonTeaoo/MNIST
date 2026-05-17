import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # Feature extraction
        self.features = nn.Sequential(

            # Conv Layer 1
            nn.Conv2d(
                # Gray 1 Channel
                in_channels=1,
                # Learn 32 Features
                out_channels=32,
                # Each Feature 3x3 size
                kernel_size=3,
                # Padding
                padding=1
            ),
            nn.ReLU(),

            # Max Pooling (Reduce Noice)
            nn.MaxPool2d(kernel_size=2),

            # Conv Layer 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            # Max Pooling
            nn.MaxPool2d(kernel_size=2)
        )

        # Classification
        self.classifier = nn.Sequential(
            # Convert to 1d vector
            nn.Flatten(),

            # 64 feature maps of size 7x7
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),

            nn.Linear(128, 10)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x