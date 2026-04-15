"""
Filtro 4 — Detección de Alteración por IA
Modelo: UnivFD — Universal Fake Detector (CVPR 2023)

Usa CLIP ViT-L/14 como extractor de features + una capa lineal de clasificación.
Generaliza a 19+ modelos generativos (StyleGAN, Stable Diffusion, DALL-E, Midjourney).

Licencia: MIT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from app.config import AI_AMBIGUOUS_HIGH, AI_AMBIGUOUS_LOW, AI_CLIP_MEAN, AI_CLIP_STD, AI_MODEL_INPUT_SIZE, weights_dir
from app.schemas import FilterResult

logger = logging.getLogger(__name__)


class UnivFDClassifier(nn.Module):
    """
    Universal Fake Detector: CLIP ViT-L/14 (congelado) + Linear(768, 1).
    Usa open_clip para descargar automáticamente los pesos de CLIP.
    """

    def __init__(self) -> None:
        super().__init__()
        import open_clip

        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        self.clip_model.eval()
        # Congelar CLIP completamente
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(768, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.clip_model.encode_image(x)
        features = features.float()
        return self.fc(features)


@dataclass
class AIAlteredDetector:
    """Filtro 4: ¿La imagen fue generada o modificada por IA?"""

    threshold: float = 0.50
    device: str = "cpu"
    _model: UnivFDClassifier | None = field(default=None, repr=False)
    _transform: transforms.Compose | None = field(default=None, repr=False)
    _use_model: bool = field(default=False, repr=False)

    def load_model(self, device: str = "cpu") -> None:
        self.device = device
        self._model = UnivFDClassifier()

        fc_weights = weights_dir() / "univfd_fc_weights.pth"
        if fc_weights.exists():
            state = torch.load(fc_weights, map_location=device, weights_only=True)
            self._model.fc.load_state_dict(state)
            logger.info("UnivFD fc weights loaded from %s", fc_weights)
        else:
            logger.warning(
                "No fc weights at %s — classifier is untrained. "
                "Download from UniversalFakeDetect repo pretrained_weights/fc_weights.pth",
                fc_weights,
            )

        self._model.to(device).eval()
        self._use_model = True
        self._transform = transforms.Compose([
            transforms.Resize(AI_MODEL_INPUT_SIZE),
            transforms.CenterCrop(AI_MODEL_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=AI_CLIP_MEAN, std=AI_CLIP_STD),
        ])

    def predict(self, image: Image.Image) -> FilterResult:
        if self._use_model and self._model is not None:
            return self._predict_model(image)
        return self._predict_heuristic(image)

    @torch.no_grad()
    def _predict_model(self, image: Image.Image) -> FilterResult:
        tensor = self._transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        logit = self._model(tensor)
        prob = torch.sigmoid(logit).item()
        # prob > 0.5 → fake
        is_ai = prob > self.threshold
        confidence = prob if is_ai else 1.0 - prob

        # Log si ambiguo (para potencial activación de DIRE en futuro)
        if AI_AMBIGUOUS_LOW < prob < AI_AMBIGUOUS_HIGH:
            logger.info("UnivFD ambiguous score %.3f — consider DIRE secondary check", prob)

        return FilterResult(
            answer="yes" if is_ai else "no",
            percentageOfConfidence=round(confidence * 100, 1),
        )

    def _predict_heuristic(self, image: Image.Image) -> FilterResult:
        """Fallback: análisis de artefactos estadísticos simples."""
        import numpy as np

        rgb = np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32)

        # GAN/diffusion tiende a producir distribuciones de color más uniformes
        channel_kurtosis = []
        for c in range(3):
            vals = rgb[:, :, c].flatten()
            mu = np.mean(vals)
            std = np.std(vals) + 1e-8
            kurt = float(np.mean(((vals - mu) / std) ** 4) - 3.0)
            channel_kurtosis.append(kurt)

        avg_kurt = float(np.mean(channel_kurtosis))
        # Imágenes reales tienen kurtosis más alta; AI-generated tiende a 0
        norm = max(0.0, min(1.0, 1.0 - (avg_kurt / 5.0)))
        is_ai = norm > self.threshold
        confidence = norm if is_ai else 1.0 - norm
        return FilterResult(
            answer="yes" if is_ai else "no",
            percentageOfConfidence=round(confidence * 100, 1),
        )
