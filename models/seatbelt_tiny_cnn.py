"""
Tiny binary CNN for seat-belt detection (bonus task).
Target: <50K parameters for ultra-fast inference.
"""

import torch
import torch.nn as nn


class SeatbeltTinyCNN(nn.Module):
    """Ultra-lightweight binary classifier for seat-belt detection."""

    def __init__(self, input_channels=1):
        super().__init__()

        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_seatbelt_model(grayscale=True):
    """Factory function to create seat-belt model."""
    input_channels = 1 if grayscale else 3
    model = SeatbeltTinyCNN(input_channels)
    print(f"Seat-belt model parameters: {model.count_parameters():,}")
    return model
