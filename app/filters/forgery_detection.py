"""
Filtro 3 — Detección de Elementos Superpuestos / Falsificación
Modelo: HiFi-IFDL (MIT license) — Hierarchical Fine-Grained Image Forgery Detection

Detecta: stickers pegados, fotos sustituidas, texto alterado, splicing,
copy-move, y cualquier manipulación a nivel de píxel.

Produce:
- Score de integridad (0-1)
- Máscara de localización (256x256) indicando regiones manipuladas
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from app.config import FORGERY_MODEL_INPUT_SIZE, FORGERY_THRESHOLD, weights_dir
from app.schemas import FilterResult

logger = logging.getLogger(__name__)


class ForgeryFeatureExtractor(nn.Module):
    """
    Feature extractor inspirado en HiFi-IFDL: rama RGB + rama frecuencial.
    Usa ResNet-18 como backbone ligero con dual-stream input.
    """

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Rama RGB
        self.rgb_conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.rgb_bn1 = backbone.bn1
        self.rgb_relu = backbone.relu
        self.rgb_pool = backbone.maxpool

        # Rama frecuencial (SRM-inspired noise filters)
        self.freq_conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.freq_bn1 = nn.BatchNorm2d(32)
        self.freq_conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False)
        self.freq_bn2 = nn.BatchNorm2d(64)
        self.freq_pool = nn.MaxPool2d(3, stride=2, padding=1)

        # Fusión
        self.merge = nn.Conv2d(128, 64, 1, bias=False)
        self.merge_bn = nn.BatchNorm2d(64)

        # Shared ResNet layers
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Clasificación
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 2),  # real vs forged
        )

        # Segmentación (localización)
        self.seg_head = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )

        # Inicializar rama RGB con pesos de ResNet
        with torch.no_grad():
            self.rgb_conv1.weight.copy_(backbone.conv1.weight)

    def _compute_noise_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Filtra residuo de ruido con kernels SRM simplificados."""
        return x - F.avg_pool2d(F.pad(x, [1, 1, 1, 1], mode="reflect"), 3, stride=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Rama RGB
        rgb = self.rgb_conv1(x)
        rgb = self.rgb_relu(self.rgb_bn1(rgb))
        rgb = self.rgb_pool(rgb)

        # Rama frecuencial
        noise = self._compute_noise_residual(x)
        freq = F.relu(self.freq_bn1(self.freq_conv1(noise)))
        freq = F.relu(self.freq_bn2(self.freq_conv2(freq)))
        freq = self.freq_pool(freq)

        # Fusión
        merged = torch.cat([rgb, freq], dim=1)
        merged = F.relu(self.merge_bn(self.merge(merged)))

        # Backbone compartido
        f1 = self.layer1(merged)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        # Clasificación
        cls_feat = self.gap(f4).flatten(1)
        cls_logits = self.classifier(cls_feat)

        # Localización
        seg_map = self.seg_head(f4)
        seg_map = F.interpolate(seg_map, size=(FORGERY_MODEL_INPUT_SIZE, FORGERY_MODEL_INPUT_SIZE),
                                mode="bilinear", align_corners=False)

        return cls_logits, seg_map


@dataclass
class ForgeryDetector:
    """Filtro 3: ¿El documento tiene elementos superpuestos o alterados?"""

    threshold: float = FORGERY_THRESHOLD
    device: str = "cpu"
    _model: ForgeryFeatureExtractor | None = field(default=None, repr=False)
    _use_model: bool = field(default=False, repr=False)

    def load_model(self, device: str = "cpu") -> None:
        self.device = device
        checkpoint = weights_dir() / "forgery_hifi.pth"
        self._model = ForgeryFeatureExtractor()
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            self._model.load_state_dict(state)
            logger.info("Forgery detector loaded from %s", checkpoint)
        else:
            logger.warning("No checkpoint at %s — using ImageNet init.", checkpoint)
        self._model.to(device).eval()
        self._use_model = True

    def predict(self, image: Image.Image) -> FilterResult:
        if self._use_model and self._model is not None:
            return self._predict_model(image)
        return self._predict_heuristic(image)

    @torch.no_grad()
    def _predict_model(self, image: Image.Image) -> FilterResult:
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((FORGERY_MODEL_INPUT_SIZE, FORGERY_MODEL_INPUT_SIZE)),
            transforms.ToTensor(),
        ])
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        cls_logits, seg_map = self._model(tensor)
        probs = torch.softmax(cls_logits, dim=-1)[0]
        forged_prob = probs[1].item()

        # Si hay segmentación, calcular % de píxeles manipulados
        if seg_map is not None:
            mask = torch.sigmoid(seg_map[0, 0])
            tampered_ratio = (mask > 0.5).float().mean().item()
            # Combinar clasificación + localización
            combined = 0.7 * forged_prob + 0.3 * tampered_ratio
        else:
            combined = forged_prob

        is_forged = combined > self.threshold
        confidence = combined if is_forged else 1.0 - combined
        return FilterResult(
            answer="yes" if is_forged else "no",
            percentageOfConfidence=round(confidence * 100, 1),
        )

    def _predict_heuristic(self, image: Image.Image) -> FilterResult:
        """ELA (Error Level Analysis) como fallback sin modelo."""
        import io

        img = image.convert("RGB").resize((256, 256))
        original = np.asarray(img, dtype=np.float32)

        # Recomprimir con calidad baja y comparar
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        buf.seek(0)
        recompressed = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32)

        diff = np.abs(original - recompressed)
        ela_score = float(np.mean(diff) / 255.0)
        # Normalizar: documentos genuinos tienen ELA uniforme, alterados tienen picos
        ela_std = float(np.std(diff) / 255.0)
        anomaly = min(1.0, ela_std * 5.0)

        combined = 0.4 * min(1.0, ela_score * 10) + 0.6 * anomaly
        is_forged = combined > self.threshold
        confidence = combined if is_forged else 1.0 - combined
        return FilterResult(
            answer="yes" if is_forged else "no",
            percentageOfConfidence=round(confidence * 100, 1),
        )
