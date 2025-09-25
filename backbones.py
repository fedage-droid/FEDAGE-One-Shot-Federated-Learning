# backbones.py

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    def __init__(self, output_dim=64):
        super().__init__()
        base = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(base.children())[:-1])  # remove fc
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class MobileNetEncoder(nn.Module):
    def __init__(self, output_dim=64):
        super().__init__()
        base = models.mobilenet_v2(pretrained=True)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1280, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class SqueezeNetEncoder(nn.Module):
    def __init__(self, output_dim=64):
        super().__init__()
        self.squeezenet = models.squeezenet1_1(pretrained=True)
        self.features = self.squeezenet.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class VGGEncoder(nn.Module):
    def __init__(self, output_dim=64):
        super().__init__()
        base = models.vgg19(pretrained=True)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(512 * 7 * 7, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def get_encoder(name, output_dim=64):
    name = name.lower()
    if name == "resnet18":
        return ResNetEncoder(output_dim)
    elif name == "mobilenet":
        return MobileNetEncoder(output_dim)
    elif name == "squeezenet":
        return SqueezeNetEncoder(output_dim)
    elif name == "vgg19":
        return VGGEncoder(output_dim)
    else:
        raise ValueError(f"Unknown encoder backbone: {name}")
