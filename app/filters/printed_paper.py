"""
Filtro 2 — Detección de Documento Impreso
Modelo: EfficientNet-B4 + análisis de frecuencia FFT (FHAG-inspired)

Detecta patrones de medios tonos (halftone) del proceso de impresión
que no existen en documentos originales laminados/plásticos.
Fine-tunear sobre DLC-2021 para producción.
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
    PRINT_FFT_BAND_HIGH,
    PRINT_FFT_BAND_LOW,
    PRINT_MODEL_INPUT_SIZE,
    SCREEN_IMAGENET_MEAN,
    SCREEN_IMAGENET_STD,
    weights_dir,
)
from app.schemas import FilterResult

logger = logging.getLogger(__name__)


def extract_fft_features(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Extrae mapa de energía espectral en banda de medios tonos.
    Input:  (B, 3, H, W)
    Output: (B, 1, H, W) — energía en banda de interés
    """
    gray = image_tensor.mean(dim=1, keepdim=True)  # (B, 1, H, W)
    fft = torch.fft.fft2(gray)
    fft_shift = torch.fft.fftshift(fft)
    magnitude = torch.abs(fft_shift)

    _, _, h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y = torch.arange(h, device=magnitude.device).float().unsqueeze(1)
    x = torch.arange(w, device=magnitude.device).float().unsqueeze(0)
    r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    band_mask = ((r >= PRINT_FFT_BAND_LOW) & (r <= PRINT_FFT_BAND_HIGH)).float()
    band_mask = band_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    return magnitude * band_mask


class PrintDetectionClassifier(nn.Module):
    """EfficientNet-B4 con canal FFT adicional para halftone detection."""

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4

        backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        old_conv = backbone.features[0][0]
        # 4 canales: RGB + FFT band energy
        new_conv = nn.Conv2d(
            4, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:4] = old_conv.weight[:, 0:1].clone()
        backbone.features[0][0] = new_conv
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1792, 2))

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        fft_band = extract_fft_features(rgb)
        # Normalizar FFT a [0,1]
        fft_max = fft_band.amax(dim=(-2, -1), keepdim=True) + 1e-8
        fft_norm = fft_band / fft_max
        x = torch.cat([rgb, fft_norm], dim=1)  # (B, 4, H, W)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


@dataclass
class PrintedPaperDetector:
    """Filtro 2: ¿El documento fue impreso en papel?"""

    threshold: float = 0.48
    device: str = "cpu"
    _model: PrintDetectionClassifier | None = field(default=None, repr=False)
    _transform: transforms.Compose | None = field(default=None, repr=False)
    _use_model: bool = field(default=False, repr=False)

    def load_model(self, device: str = "cpu") -> None:
        self.device = device
        checkpoint = weights_dir() / "print_detection_effnet.pth"
        self._model = PrintDetectionClassifier()
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            self._model.load_state_dict(state)
            logger.info("Print detection model loaded from %s", checkpoint)
        else:
            logger.warning("No checkpoint at %s — using ImageNet init.", checkpoint)
        self._model.to(device).eval()
        self._use_model = True
        self._transform = transforms.Compose([
            transforms.Resize(PRINT_MODEL_INPUT_SIZE),
            transforms.CenterCrop(PRINT_MODEL_INPUT_SIZE),
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
        print_prob = probs[1].item()
        is_printed = print_prob > self.threshold
        confidence = print_prob if is_printed else 1.0 - print_prob
        return FilterResult(
            answer="yes" if is_printed else "no",
            percentageOfConfidence=round(confidence * 100, 1),
        )

    def _predict_heuristic(self, image: Image.Image) -> FilterResult:
        gray = np.asarray(image.convert("L").resize((380, 380)), dtype=np.float32)
        fft = np.fft.fft2(gray)
        mag = np.abs(np.fft.fftshift(fft))
        h, w = mag.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        band = ((r >= PRINT_FFT_BAND_LOW) & (r <= PRINT_FFT_BAND_HIGH)).astype(np.float32)
        band_energy = float(np.sum(mag * band) / (np.sum(band) + 1e-8))
        total_energy = float(np.sum(mag) / (h * w))
        ratio = band_energy / (total_energy + 1e-8)
        # Textura de gradiente local
        gy, gx = np.gradient(gray)
        texture = float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))
        tex_norm = max(0.0, min(1.0, (texture - 8.0) / 35.0))
        freq_norm = max(0.0, min(1.0, (ratio - 0.5) / 2.0))
        combined = 0.6 * freq_norm + 0.4 * tex_norm
        is_printed = combined > self.threshold
        confidence = combined if is_printed else 1.0 - combined
        return FilterResult(
            answer="yes" if is_printed else "no",
            percentageOfConfidence=round(confidence * 100, 1),
        )
