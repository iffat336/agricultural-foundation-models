from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SyntheticDataConfig:
    num_classes: int = 3
    image_channels: int = 6
    image_size: int = 16
    weather_steps: int = 12
    weather_features: int = 5
    geo_features: int = 4


class SyntheticAgricultureDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        split: str,
        seed: int = 42,
        config: SyntheticDataConfig | None = None,
    ) -> None:
        self.config = config or SyntheticDataConfig()
        self.num_samples = num_samples
        self.split = split
        self.seed = seed

        rng = np.random.default_rng(seed + self._split_offset(split))
        labels = rng.integers(0, self.config.num_classes, size=num_samples)
        images = []
        weather = []
        geo = []

        for label in labels:
            images.append(self._generate_image(rng, label))
            weather.append(self._generate_weather(rng, label))
            geo.append(self._generate_geo(rng, label))

        self.images = torch.tensor(np.stack(images), dtype=torch.float32)
        self.weather = torch.tensor(np.stack(weather), dtype=torch.float32)
        self.geo = torch.tensor(np.stack(geo), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "image": self.images[index],
            "weather": self.weather[index],
            "geo": self.geo[index],
            "label": self.labels[index],
        }

    def _split_offset(self, split: str) -> int:
        offsets = {
            "train": 0,
            "val": 11,
            "region_shift": 23,
            "year_shift": 37,
            "missing_modality": 51,
        }
        return offsets.get(split, 7)

    def _generate_image(self, rng: np.random.Generator, label: int) -> np.ndarray:
        cfg = self.config
        image = rng.normal(0.0, 0.35, size=(cfg.image_channels, cfg.image_size, cfg.image_size)).astype(np.float32)

        if label == 0:
            image[0, :8, :8] += 1.8
            image[1, 8:, 8:] += 0.8
        elif label == 1:
            image[2, 4:12, 4:12] += 1.6
            image[3, :, 2:4] += 0.7
        else:
            image[4, :, :] += 0.5
            image[5, 6:10, :] += 1.3

        if self.split == "region_shift":
            image += 0.25
            image *= 1.08
        elif self.split == "year_shift":
            image += rng.normal(0.0, 0.15, size=image.shape).astype(np.float32)
        elif self.split == "missing_modality":
            image *= 0.95

        return image

    def _generate_weather(self, rng: np.random.Generator, label: int) -> np.ndarray:
        cfg = self.config
        base = rng.normal(0.0, 0.25, size=(cfg.weather_steps, cfg.weather_features)).astype(np.float32)
        seasonal_curve = np.linspace(-1.0, 1.0, cfg.weather_steps, dtype=np.float32)[:, None]
        label_offsets = {
            0: np.array([0.6, -0.3, 0.3, 0.1, 0.0], dtype=np.float32),
            1: np.array([-0.1, 0.5, -0.2, 0.4, 0.2], dtype=np.float32),
            2: np.array([0.2, 0.2, 0.6, -0.2, 0.4], dtype=np.float32),
        }
        weather = base + seasonal_curve * label_offsets[label]

        if self.split == "region_shift":
            weather += 0.2
        elif self.split == "year_shift":
            weather += np.linspace(0.3, -0.2, cfg.weather_steps, dtype=np.float32)[:, None]
        elif self.split == "missing_modality":
            weather[:, :2] = 0.0

        return weather

    def _generate_geo(self, rng: np.random.Generator, label: int) -> np.ndarray:
        geo = rng.normal(0.0, 0.3, size=(self.config.geo_features,)).astype(np.float32)
        geo += np.eye(self.config.num_classes, self.config.geo_features, dtype=np.float32)[label] * 0.9

        if self.split == "region_shift":
            geo += np.array([0.4, -0.2, 0.1, 0.0], dtype=np.float32)
        elif self.split == "year_shift":
            geo += np.array([0.0, 0.3, -0.2, 0.1], dtype=np.float32)

        return geo


def build_synthetic_splits(seed: int = 42) -> dict[str, SyntheticAgricultureDataset]:
    return {
        "train": SyntheticAgricultureDataset(num_samples=768, split="train", seed=seed),
        "val": SyntheticAgricultureDataset(num_samples=192, split="val", seed=seed),
        "region_shift": SyntheticAgricultureDataset(num_samples=256, split="region_shift", seed=seed),
        "year_shift": SyntheticAgricultureDataset(num_samples=256, split="year_shift", seed=seed),
        "missing_modality": SyntheticAgricultureDataset(num_samples=256, split="missing_modality", seed=seed),
    }
