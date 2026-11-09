"""Minimal reusable utilities for PathMNIST."""
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def classification_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

def corrupt_batch(x, kind="gaussian_noise", severity=0.1):
    """x is a float tensor in [0,1]. Synthetic test-time stress tests only."""
    if kind == "gaussian_noise":
        x = x + torch.randn_like(x) * severity
    elif kind == "brightness":
        x = x * (1.0 + severity)
    else:
        raise ValueError("kind must be gaussian_noise or brightness")
    return x.clamp(0, 1)

class SmallCNN(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.num_classes = num_classes
        # PathMNIST images are 3x28x28
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Out: 16 x 14 x 14

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Out: 32 x 7 x 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = SmallCNN().to(device)
print(model)
