"""
Filtro 1 — Detección de Captura de Pantalla
Modelo: EfficientNet-B4 + Chromaticity Map Adapter (inspirado en CMA CVPR 2024)

Detecta artefactos de recaptura: moiré, sub-pixel misalignment,
chromaticity shifts, doble gamma.

Usa EfficientNet-B4 (Apache 2.0) con 6 canales de entrada (RGB + chromaticity map)
para mantener licencia comercial. Fine-tunear sobre SIDTD + DLC-2021.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from app.config import (
    SCREEN_IMAGENET_MEAN,
    SCREEN_IMAGENET_STD,
    SCREEN_MODEL_INPUT_SIZE,
    weights_dir,
)
from app.schemas import FilterResult

logger = logging.getLogger(__name__)


def compute_chromaticity_map(x: torch.Tensor) -> torch.Tensor:
    """
    Chromaticity Map (IIC) — convierte RGB a coordenadas cromáticas normalizadas.
    Input:  (B, 3, H, W) rango [0, 1]
    Output: (B, 3, H, W) z-score normalizado
    """
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    total = r + g + b + 1e-8
    c_r, c_g, c_b = r / total, g / total, b / total
    c_stack = torch.cat([c_r, c_g, c_b], dim=1)
    mean = c_stack.mean(dim=1, keepdim=True)
    std = c_stack.std(dim=1, keepdim=True) + 1e-8
    return (c_stack - mean) / std


class CMAScreenClassifier(nn.Module):
    """EfficientNet-B4 con 6 canales (RGB + chromaticity) para screen recapture."""

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4

        backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        old_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            6, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:] = old_conv.weight.clone()
        backbone.features[0][0] = new_conv
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1792, 2))

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        chrom = compute_chromaticity_map(rgb)
        x = torch.cat([rgb, chrom], dim=1)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


@dataclass
class ScreenCaptureDetector:
    """Filtro 1: ¿La foto fue tomada desde una pantalla?"""

    threshold: float = 0.50
    device: str = "cpu"
    _model: CMAScreenClassifier | None = field(default=None, repr=False)
    _transform: transforms.Compose | None = field(default=None, repr=False)
    _use_model: bool = field(default=False, repr=False)

    def load_model(self, device: str = "cpu") -> None:
        self.device = device
        checkpoint = weights_dir() / "screen_capture_cma.pth"
        self._model = CMAScreenClassifier()
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            self._model.load_state_dict(state)
            logger.info("Screen capture model loaded from %s", checkpoint)
        else:
            logger.warning("No checkpoint at %s — using ImageNet init.", checkpoint)
        self._model.to(device).eval()
        self._use_model = True
        self._transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(SCREEN_MODEL_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=SCREEN_IMAGENET_MEAN, std=SCREEN_IMAGENET_STD),
        ])

    def predict(self, image: Image.Image) -> FilterResult:
        if self._use_model and self._model is not None:
            return self._predict_model(image)
        return self._predict_heuristic(image)

    @torch.no_grad()
    def _predict_model(self, image: Image.Image) -> FilterResult:
        tensor = self._transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self._model(tensor)
        probs = torch.softmax(logits, dim=-1)[0]
        screen_prob = probs[1].item()
        is_screen = screen_prob > self.threshold
        confidence = screen_prob if is_screen else 1.0 - screen_prob
        return FilterResult(
            answer="yes" if is_screen else "no",
            percentageOfConfidence=round(confidence * 100, 1),
        )

    def _predict_heuristic(self, image: Image.Image) -> FilterResult:
        rgb = np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        chrom = compute_chromaticity_map(tensor)[0].numpy()
        chrom_var = float(np.var(chrom))
        gray_chrom = np.mean(chrom, axis=0)
        fft = np.fft.fft2(gray_chrom)
        mag = np.abs(np.fft.fftshift(fft))
        h, w = mag.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        mid_mask = ((r > 20) & (r < 80)).astype(np.float32)
        mid_energy = float(np.sum(mag * mid_mask) / (np.sum(mid_mask) + 1e-8))
        total_energy = float(np.sum(mag) / (h * w))
        ratio = mid_energy / (total_energy + 1e-8)
        norm = max(0.0, min(1.0, (ratio - 0.8) / 1.5))
        combined = 0.5 * norm + 0.5 * min(1.0, chrom_var / 0.15)
        is_screen = combined > self.threshold
        confidence = combined if is_screen else 1.0 - combined
        return FilterResult(
            answer="yes" if is_screen else "no",
            percentageOfConfidence=round(confidence * 100, 1),
        )
