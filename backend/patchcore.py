"""PatchCore-style inference using pretrained torchvision feature maps."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet18_Weights, Wide_ResNet50_2_Weights, resnet18, wide_resnet50_2

from . import config


class PatchCore:
    """Build and query a compact memory bank of normal image patches."""

    def __init__(self, bank_path: Path | None = None) -> None:
        self.device = torch.device("cpu")
        self.bank_path = bank_path or config.DATA_DIR / "memory_bank.pt"
        self.backbone_name = config.BACKBONE
        self.model = self._load_backbone()
        self.memory_bank: torch.Tensor | None = None
        self.grid_size: tuple[int, int] | None = None
        self.category = config.CATEGORY
        self.calibration: dict[str, float] = {}
        if self.bank_path.exists():
            self.load()

    def _load_backbone(self) -> torch.nn.Module:
        """Load a pretrained feature extractor once, with classifier removed."""
        if self.backbone_name == "resnet18":
            network = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif self.backbone_name == "wide_resnet50_2":
            network = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
        network.fc = torch.nn.Identity()
        network.eval().to(self.device)
        return network

    @staticmethod
    def _as_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
        """Convert a supported image to an RGB uint8 array."""
        if isinstance(image, Image.Image):
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
        array = np.asarray(image)
        if array.ndim == 2:
            array = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError("Image must have 1, 3, or 4 channels")
        if array.shape[2] == 4:
            array = array[:, :, :3]
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array

    def _preprocess(self, image: Image.Image | np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        """Resize, center-crop, and normalize an RGB image."""
        array = self._as_rgb_array(image)
        original_size = (array.shape[0], array.shape[1])
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
        height, width = array.shape[:2]
        scale = 256.0 / min(height, width)
        resized_height, resized_width = max(224, round(height * scale)), max(224, round(width * scale))
        tensor = F.interpolate(tensor, size=(resized_height, resized_width), mode="bilinear", align_corners=False)
        top = (resized_height - config.INPUT_SIZE) // 2
        left = (resized_width - config.INPUT_SIZE) // 2
        tensor = tensor[:, :, top : top + config.INPUT_SIZE, left : left + config.INPUT_SIZE]
        mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        return (tensor - mean) / std, original_size

    @torch.no_grad()
    def extract_features(self, image: Image.Image | np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
        """Return concatenated layer2/layer3 patch descriptors and their grid."""
        tensor, _ = self._preprocess(image)
        network = self.model
        value = network.maxpool(network.relu(network.bn1(network.conv1(tensor))))
        value = network.layer1(value)
        layer2 = network.layer2(value)
        layer3 = network.layer3(layer2)
        layer3 = F.interpolate(layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False)
        pooled2 = F.avg_pool2d(layer2, kernel_size=3, stride=1, padding=1)
        pooled3 = F.avg_pool2d(layer3, kernel_size=3, stride=1, padding=1)
        descriptors = torch.cat((pooled2, pooled3), dim=1)
        grid = (int(descriptors.shape[-2]), int(descriptors.shape[-1]))
        patches = descriptors.squeeze(0).permute(1, 2, 0).reshape(-1, descriptors.shape[1]).contiguous()
        return patches.cpu(), grid

    @staticmethod
    def _subsample(patches: torch.Tensor, limit: int) -> torch.Tensor:
        """Keep a reproducible random subset no larger than limit."""
        if patches.shape[0] <= limit:
            return patches.contiguous()
        indices = torch.randperm(patches.shape[0])[:limit]
        return patches[indices].contiguous()

    def fit(self, images: list[Image.Image | np.ndarray]) -> dict[str, Any]:
        """Create a memory bank from normal images without parameter training."""
        if not images:
            raise ValueError("At least one good image is required")
        image_features: list[torch.Tensor] = []
        grids: list[tuple[int, int]] = []
        for image in images:
            features, grid = self.extract_features(image)
            image_features.append(features)
            grids.append(grid)
        if len(set(grids)) != 1:
            raise ValueError("Good images must produce the same feature grid")
        self.grid_size = grids[0]
        self.memory_bank = self._subsample(torch.cat(image_features, dim=0), config.MAX_BANK_PATCHES)
        self.category = config.CATEGORY
        self.save()
        return {"images": len(images), "patches": int(self.memory_bank.shape[0]), "grid_size": list(self.grid_size)}

    def _nearest_distances(self, queries: torch.Tensor, bank: torch.Tensor | None = None) -> torch.Tensor:
        """Calculate nearest memory-bank distances in bounded query chunks."""
        reference = bank if bank is not None else self.memory_bank
        if reference is None or reference.numel() == 0:
            raise RuntimeError("Fit the model first")
        distances: list[torch.Tensor] = []
        reference = reference.to(dtype=torch.float32, device=self.device)
        for start in range(0, queries.shape[0], config.DISTANCE_CHUNK_SIZE):
            batch = queries[start : start + config.DISTANCE_CHUNK_SIZE].to(dtype=torch.float32, device=self.device)
            distances.append(torch.cdist(batch, reference).min(dim=1).values.cpu())
        return torch.cat(distances)

    def _score_with_bank(
        self, image: Image.Image | np.ndarray, bank: torch.Tensor
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        """Score an image against a specified bank, used for leave-one-out calibration."""
        features, grid = self.extract_features(image)
        patch_distances = self._nearest_distances(features, bank).reshape(1, 1, *grid)
        array = self._as_rgb_array(image)
        height, width = array.shape[:2]
        anomaly = F.interpolate(patch_distances, size=(height, width), mode="bilinear", align_corners=False)
        anomaly_map = anomaly.squeeze().numpy().astype(np.float32)
        anomaly_map = cv2.GaussianBlur(anomaly_map, (0, 0), sigmaX=config.GAUSSIAN_SIGMA)
        maximum = float(anomaly_map.max())
        y, x = np.unravel_index(int(anomaly_map.argmax()), anomaly_map.shape)
        return anomaly_map, maximum, (x / max(width - 1, 1), y / max(height - 1, 1))

    def score(self, image: Image.Image | np.ndarray) -> dict[str, Any]:
        """Return a full-resolution anomaly map, maximum score, and hotspot."""
        if self.memory_bank is None:
            raise RuntimeError("Fit the model first")
        anomaly_map, maximum, hotspot = self._score_with_bank(image, self.memory_bank)
        return {"anomaly_map": anomaly_map, "max_score": maximum, "hotspot": hotspot}

    def add_to_bank(self, image: Image.Image | np.ndarray) -> int:
        """Add confirmed-normal patches and persist the updated bank."""
        features, grid = self.extract_features(image)
        if self.memory_bank is None:
            self.memory_bank = self._subsample(features, config.MAX_BANK_PATCHES)
            self.grid_size = grid
        else:
            self.memory_bank = self._subsample(torch.cat((self.memory_bank, features), dim=0), config.MAX_BANK_PATCHES)
        self.save()
        return int(self.memory_bank.shape[0])

    def calibrate(self, good_images: list[Image.Image | np.ndarray]) -> dict[str, float]:
        """Estimate normal-score spread with leave-one-image-out comparisons."""
        if len(good_images) < 2:
            raise ValueError("At least two good images are needed for calibration")
        feature_sets = [self.extract_features(image)[0] for image in good_images]
        scores: list[float] = []
        for held_out, image in enumerate(good_images):
            other_patches = torch.cat([patches for index, patches in enumerate(feature_sets) if index != held_out])
            other_bank = self._subsample(other_patches, config.MAX_BANK_PATCHES)
            scores.append(self._score_with_bank(image, other_bank)[1])
        values = np.asarray(scores, dtype=np.float64)
        self.calibration = {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
        self.save()
        return self.calibration.copy()

    def confidence_pct(self, score: float) -> float:
        """Map scores to a bounded 0-100 confidence relative to calibration."""
        low = self.calibration.get("min", 0.0)
        high = self.calibration.get("max", low)
        if high <= low:
            return 0.0 if score > high else 100.0
        return float(np.clip(100.0 * (high - score) / (high - low), 0.0, 100.0))

    def save(self) -> None:
        """Persist bank and metadata to the configured local path."""
        if self.memory_bank is None:
            return
        self.bank_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "memory_bank": self.memory_bank.cpu(),
                "grid_size": self.grid_size,
                "category": self.category,
                "backbone": self.backbone_name,
                "calibration": self.calibration,
            },
            self.bank_path,
        )

    def load(self) -> None:
        """Load the saved CPU bank and calibration metadata."""
        saved = torch.load(self.bank_path, map_location="cpu", weights_only=False)
        if saved.get("backbone") != self.backbone_name:
            raise ValueError("Saved memory bank uses a different backbone")
        self.memory_bank = saved["memory_bank"].float()
        self.grid_size = tuple(saved["grid_size"]) if saved.get("grid_size") else None
        self.category = str(saved.get("category", config.CATEGORY))
        self.calibration = {key: float(value) for key, value in saved.get("calibration", {}).items()}
