"""
Small CNN for emotion classification with depthwise separable convolutions.
Target: <0.5M parameters, optimized for mobile/edge deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution block."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EmotionSmallCNN(nn.Module):
    """Lightweight CNN for 7-class emotion recognition."""

    def __init__(self, num_classes=7, input_channels=1, dropout=0.2):
        super().__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 64x64 -> 32x32
        )

        # Depthwise separable blocks
        self.block1 = nn.Sequential(
            DepthwiseSeparableConv(32, 64),
            nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        )

        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(64, 128),
            nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        )

        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(128, 128),
            nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_emotion_model(num_classes=7, grayscale=True, pretrained=False):
    """Factory function to create emotion model."""
    input_channels = 1 if grayscale else 3
    model = EmotionSmallCNN(num_classes, input_channels)
    print(f"Model parameters: {model.count_parameters():,}")
    return model
