from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SyntheticDataConfig:
    num_classes: int = 3
    image_channels: int = 6
    image_size: int = 20
    weather_steps: int = 16
    weather_features: int = 6
    geo_features: int = 5
    management_features: int = 4
    text_features: int = 8


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

        images = []
        weather = []
        geo = []
        management = []
        text = []
        labels = []
        physics_targets = []

        for _ in range(num_samples):
            sample = self._generate_sample(rng)
            images.append(sample["image"])
            weather.append(sample["weather"])
            geo.append(sample["geo"])
            management.append(sample["management"])
            text.append(sample["text"])
            labels.append(sample["label"])
            physics_targets.append(sample["physics_target"])

        self.images = torch.tensor(np.stack(images), dtype=torch.float32)
        self.weather = torch.tensor(np.stack(weather), dtype=torch.float32)
        self.geo = torch.tensor(np.stack(geo), dtype=torch.float32)
        self.management = torch.tensor(np.stack(management), dtype=torch.float32)
        self.text = torch.tensor(np.stack(text), dtype=torch.float32)
        self.labels = torch.tensor(np.array(labels), dtype=torch.long)
        self.physics_targets = torch.tensor(np.array(physics_targets)[:, None], dtype=torch.float32)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "image": self.images[index],
            "weather": self.weather[index],
            "geo": self.geo[index],
            "management": self.management[index],
            "text": self.text[index],
            "physics_target": self.physics_targets[index],
            "label": self.labels[index],
        }

    def _split_offset(self, split: str) -> int:
        offsets = {
            "train": 0,
            "val": 11,
            "region_shift": 23,
            "year_shift": 37,
            "climate_stress": 51,
            "sensor_shift": 67,
            "missing_modality": 83,
        }
        return offsets.get(split, 7)

    def _generate_sample(self, rng: np.random.Generator) -> dict[str, np.ndarray | int | float]:
        label = int(rng.integers(0, self.config.num_classes))
        region_id = int(rng.integers(0, 3))
        management_level = float(rng.uniform(0.2, 1.0))
        soil_quality = float(rng.uniform(0.1, 1.0))
        climate_stress = float(rng.uniform(0.0, 0.45))
        season_index = float(rng.uniform(-1.0, 1.0))

        if self.split == "region_shift":
            region_id = 2
        if self.split == "year_shift":
            season_index = 0.95
        if self.split == "climate_stress":
            climate_stress = float(rng.uniform(0.45, 0.9))

        weather = self._generate_weather(rng, label, climate_stress, season_index)
        geo = self._generate_geo(rng, label, region_id, soil_quality)
        management = self._generate_management(rng, management_level, climate_stress, soil_quality)
        text = self._generate_text(rng, label, climate_stress, management_level, region_id)
        physics_target = self._compute_physics_target(weather, management, soil_quality, climate_stress)
        image = self._generate_image(rng, label, weather, climate_stress, region_id)

        if self.split == "sensor_shift":
            image = image * 0.84 + rng.normal(0.0, 0.16, size=image.shape).astype(np.float32)
        if self.split == "missing_modality":
            weather[:, :3] = 0.0
            text *= 0.35

        return {
            "image": image.astype(np.float32),
            "weather": weather.astype(np.float32),
            "geo": geo.astype(np.float32),
            "management": management.astype(np.float32),
            "text": text.astype(np.float32),
            "label": label,
            "physics_target": np.float32(physics_target),
        }

    def _generate_weather(
        self,
        rng: np.random.Generator,
        label: int,
        climate_stress: float,
        season_index: float,
    ) -> np.ndarray:
        cfg = self.config
        t = np.linspace(0.0, 1.0, cfg.weather_steps, dtype=np.float32)
        phase = np.array(
            [
                0.15 + 0.08 * label,
                0.65 - 0.05 * label,
                0.35 + 0.04 * label,
                0.55,
                0.45,
                0.25 + 0.03 * label,
            ],
            dtype=np.float32,
        )
        amplitude = np.array(
            [
                0.7 + 0.15 * label,
                0.8 - 0.1 * label,
                0.4 + 0.1 * label,
                0.5,
                0.35,
                0.45,
            ],
            dtype=np.float32,
        )
        signals = []
        for idx in range(cfg.weather_features):
            curve = np.sin(2 * np.pi * (t + phase[idx])) * amplitude[idx]
            trend = (season_index * 0.25) * (2 * t - 1)
            signals.append(curve + trend)
        weather = np.stack(signals, axis=1)
        weather += rng.normal(0.0, 0.12, size=weather.shape).astype(np.float32)

        # temperature, rain, radiation, humidity, gdd-like trend, wind proxy
        weather[:, 0] += 0.65 - climate_stress * 0.25
        weather[:, 1] += 0.25 - climate_stress * 0.6
        weather[:, 2] += 0.55 - climate_stress * 0.15
        weather[:, 3] += 0.4 + climate_stress * 0.25
        weather[:, 4] += 0.5 - climate_stress * 0.35
        weather[:, 5] += 0.2 + climate_stress * 0.2
        return weather

    def _generate_geo(
        self,
        rng: np.random.Generator,
        label: int,
        region_id: int,
        soil_quality: float,
    ) -> np.ndarray:
        geo = rng.normal(0.0, 0.12, size=(self.config.geo_features,)).astype(np.float32)
        geo += np.array(
            [
                -0.9 + region_id * 0.9,
                soil_quality,
                0.25 * label,
                0.1 + 0.2 * region_id,
                0.35 * (label == region_id % self.config.num_classes),
            ],
            dtype=np.float32,
        )
        return geo

    def _generate_management(
        self,
        rng: np.random.Generator,
        management_level: float,
        climate_stress: float,
        soil_quality: float,
    ) -> np.ndarray:
        management = np.array(
            [
                management_level,
                0.65 * management_level + 0.35 * soil_quality,
                0.55 * (1.0 - climate_stress) + 0.2 * management_level,
                0.45 * soil_quality + 0.15 * management_level,
            ],
            dtype=np.float32,
        )
        management += rng.normal(0.0, 0.05, size=management.shape).astype(np.float32)
        return management

    def _generate_text(
        self,
        rng: np.random.Generator,
        label: int,
        climate_stress: float,
        management_level: float,
        region_id: int,
    ) -> np.ndarray:
        prototype = np.zeros((self.config.text_features,), dtype=np.float32)
        prototype[label] = 1.0
        prototype[3] = 1.0 - climate_stress
        prototype[4] = climate_stress
        prototype[5] = management_level
        prototype[6] = region_id / 2.0
        prototype[7] = 0.3 + 0.2 * label
        prototype += rng.normal(0.0, 0.06, size=prototype.shape).astype(np.float32)
        return prototype

    def _compute_physics_target(
        self,
        weather: np.ndarray,
        management: np.ndarray,
        soil_quality: float,
        climate_stress: float,
    ) -> float:
        temp_signal = float(weather[:, 0].mean())
        rain_signal = float(weather[:, 1].mean())
        radiation_signal = float(weather[:, 2].mean())
        gdd_signal = float(weather[:, 4].mean())
        management_signal = float(management.mean())
        yield_proxy = (
            0.28 * temp_signal
            + 0.18 * rain_signal
            + 0.24 * radiation_signal
            + 0.22 * gdd_signal
            + 0.34 * soil_quality
            + 0.32 * management_signal
            - 0.55 * climate_stress
        )
        return np.clip(yield_proxy, -1.0, 1.5)

    def _generate_image(
        self,
        rng: np.random.Generator,
        label: int,
        weather: np.ndarray,
        climate_stress: float,
        region_id: int,
    ) -> np.ndarray:
        cfg = self.config
        image = rng.normal(0.0, 0.18, size=(cfg.image_channels, cfg.image_size, cfg.image_size)).astype(np.float32)

        vigor = float(weather[:, 2].mean() + weather[:, 4].mean())
        moisture = float(weather[:, 1].mean() - climate_stress)

        # The same crop can look visually different across regions, which makes image-only learning less reliable.
        visual_label = (label + region_id) % cfg.num_classes

        image[0, 3:15, 3:15] += 0.10 + vigor * 0.05
        image[1, :, 8:10] += 0.06 + moisture * 0.03

        if visual_label == 0:
            image[0, 2:14, 2:14] += 0.42 + vigor * 0.10
            image[1, 10:18, 3:11] += 0.16 + moisture * 0.07
        elif visual_label == 1:
            image[2, 4:17, 4:17] += 0.38 + vigor * 0.09
            image[3, :, 5:7] += 0.18 + 0.04 * region_id
        else:
            image[4, :, :] += 0.14 + vigor * 0.08
            image[5, 6:12, :] += 0.34 - climate_stress * 0.10

        image += rng.normal(0.0, 0.08, size=image.shape).astype(np.float32)
        image[0] *= 1.0 - climate_stress * 0.25
        image[2] *= 1.0 - climate_stress * 0.16
        image += region_id * 0.05
        return image


def build_synthetic_splits(seed: int = 42) -> dict[str, SyntheticAgricultureDataset]:
    return {
        "train": SyntheticAgricultureDataset(num_samples=1024, split="train", seed=seed),
        "val": SyntheticAgricultureDataset(num_samples=256, split="val", seed=seed),
        "region_shift": SyntheticAgricultureDataset(num_samples=320, split="region_shift", seed=seed),
        "year_shift": SyntheticAgricultureDataset(num_samples=320, split="year_shift", seed=seed),
        "climate_stress": SyntheticAgricultureDataset(num_samples=320, split="climate_stress", seed=seed),
        "sensor_shift": SyntheticAgricultureDataset(num_samples=320, split="sensor_shift", seed=seed),
        "missing_modality": SyntheticAgricultureDataset(num_samples=320, split="missing_modality", seed=seed),
    }
