import torch
import torch.nn as nn


class BaselineMLP(nn.Module):

    def __init__(self):
        super(BaselineMLP, self).__init__()

        self.network = nn.Sequential(
            # Input to Hidden Layer
            nn.Linear(784, 256),
            # ReLU transform
            nn.ReLU(),
            # Hidden to Hidden Layer
            nn.Linear(256, 128),
            # ReLU transform
            nn.ReLU(),
            # Hidden to Output Layer (Logits for Classification)
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)