from __future__ import annotations

import torch
from torch import nn


class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int = 6, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class WeatherEncoder(nn.Module):
    def __init__(self, weather_steps: int = 12, weather_features: int = 5, hidden_dim: int = 32) -> None:
        super().__init__()
        in_dim = weather_steps * weather_features
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, weather: torch.Tensor) -> torch.Tensor:
        return self.net(weather)


class GeoEncoder(nn.Module):
    def __init__(self, geo_features: int = 4, hidden_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(geo_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, geo: torch.Tensor) -> torch.Tensor:
        return self.net(geo)


class ImageOnlyClassifier(nn.Module):
    def __init__(self, in_channels: int = 6, num_classes: int = 3) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(in_channels=in_channels, hidden_dim=64)
        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, image: torch.Tensor, weather: torch.Tensor | None = None, geo: torch.Tensor | None = None) -> torch.Tensor:
        image_repr = self.image_encoder(image)
        return self.head(image_repr)


class MultimodalFusionClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        weather_steps: int = 12,
        weather_features: int = 5,
        geo_features: int = 4,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(in_channels=in_channels, hidden_dim=64)
        self.weather_encoder = WeatherEncoder(weather_steps=weather_steps, weather_features=weather_features, hidden_dim=32)
        self.geo_encoder = GeoEncoder(geo_features=geo_features, hidden_dim=16)
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32 + 16, 96),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(96, num_classes),
        )

    def forward(self, image: torch.Tensor, weather: torch.Tensor, geo: torch.Tensor) -> torch.Tensor:
        image_repr = self.image_encoder(image)
        weather_repr = self.weather_encoder(weather)
        geo_repr = self.geo_encoder(geo)
        joint = torch.cat([image_repr, weather_repr, geo_repr], dim=1)
        return self.fusion(joint)
