import torch
import torch.nn as nn
from torchvision import models


def get_resnet50(num_classes=2, device='cpu'):
    """
    ResNet50 pretrained on ImageNet.
    We freeze all layers except the final classifier —
    so training is fast even on CPU.
    """
    # Load pretrained ResNet50
    model = models.resnet50(weights='IMAGENET1K_V1')

    # Freeze all layers — keep pretrained features
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer for our task
    in_features = model.fc.in_features   # 2048
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )

    # Modify first conv layer to accept grayscale (1 channel)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                            stride=2, padding=3, bias=False)

    return model.to(device)


def get_mobilenet(num_classes=2, device='cpu'):
    """
    MobileNetV2 — lighter than ResNet50, faster on CPU.
    Good alternative if ResNet50 is too slow.
    """
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )

    # Modify first layer for grayscale
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3,
                                      stride=2, padding=1, bias=False)

    return model.to(device)