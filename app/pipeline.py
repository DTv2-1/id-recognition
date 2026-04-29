"""
ADAMO ID — Pipeline de verificación de documentos.
Motor único: Gemini Vision con prompt unificado (1 llamada por imagen).
"""

from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass, field
from uuid import uuid4

from PIL import Image

from app.config import ALLOWED_FORMATS, MAX_DIMENSION, MAX_IMAGE_BYTES, VERDICT_WEIGHTS
from app.filters.gemini_engine import GeminiEngine
from app.schemas import (
    FilterResult,
    RiskLevel,
    Verdict,
    VerifyFiltersResponse,
    VerifyRequest,
    VerifyResponse,
)

logger = logging.getLogger(__name__)


@dataclass
class VerificationPipeline:
    """Pipeline con Gemini como único motor."""

    gemini: GeminiEngine = field(default_factory=GeminiEngine)

    def load_models(self, device: str | None = None) -> None:  # noqa: ARG002
        """Inicializa el cliente de Gemini."""
        self.gemini.initialize()
        if not self.gemini.available:
            logger.error("Gemini not available — set GEMINI_API_KEY.")

    def verify(self, payload: VerifyRequest) -> VerifyResponse:
        """Ejecuta los 5 filtros (vía 1 llamada unificada) y produce el veredicto."""
        start = time.perf_counter()
        image = self._decode_image(payload.image)

        results = self.gemini.analyze_all(image)

        # Si Gemini falló por completo, devolvemos un FilterResult neutro.
        def _or_neutral(r: FilterResult | None) -> FilterResult:
            return r if r is not None else FilterResult(answer="no", percentageOfConfidence=50.0)

        screen   = _or_neutral(results.get("screen_capture"))
        printed  = _or_neutral(results.get("printed_paper"))
        superim  = _or_neutral(results.get("superimposed_elements"))
        ai_alt   = _or_neutral(results.get("ai_altered"))

        # "liveness" = filtro 5 = score consolidado de autenticidad.
        # Es "yes" si CUALQUIER detector 1–4 disparó fraude (con la
        # confianza de ese detector). En caso contrario, "no" con la
        # confianza mínima de los 4 (= la más débil de las garantías).
        liveness = self._consolidated_liveness([screen, printed, superim, ai_alt])

        filters = VerifyFiltersResponse(
            screen_capture=screen,
            printed_paper=printed,
            superimposed_elements=superim,
            ai_altered=ai_alt,
            liveness=liveness,
        )
        verdict = self._aggregate_verdict(filters, payload.options.confidence_threshold)
        request_id = uuid4()

        logger.info(
            "request_id=%s verdict=%s confidence=%.1f risk=%s",
            request_id, verdict.is_authentic, verdict.overall_confidence, verdict.risk_level,
        )

        return VerifyResponse(
            request_id=request_id,
            processing_time_ms=int((time.perf_counter() - start) * 1000),
            verdict=verdict,
            filters=filters,
        )

    # ── Helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _decode_image(image_b64: str) -> Image.Image:
        """Decodifica base64, valida formato, tamaño y dimensiones."""
        encoded = image_b64.split(",")[-1]
        raw = base64.b64decode(encoded, validate=True)

        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError(f"Image exceeds {MAX_IMAGE_BYTES // (1024 * 1024)}MB limit")

        image = Image.open(io.BytesIO(raw))
        fmt = (image.format or "").upper()
        if fmt not in ALLOWED_FORMATS:
            raise ValueError(f"Unsupported format '{fmt}'. Allowed: {ALLOWED_FORMATS}")

        w, h = image.size
        if w > MAX_DIMENSION or h > MAX_DIMENSION:
            raise ValueError(f"Image dimensions {w}x{h} exceed {MAX_DIMENSION}x{MAX_DIMENSION}")

        return image.convert("RGB")

    @staticmethod
    def _consolidated_liveness(detectors: list[FilterResult]) -> FilterResult:
        """Filtro 5 ("liveness") = score global de autenticidad.

        - answer="yes" si CUALQUIER detector 1–4 disparó fraude.
          La confianza es la máxima de los detectores que dispararon.
        - answer="no" si todos dijeron que la imagen es auténtica.
          La confianza es la mínima de las garantías (= la más débil).
        """
        attacks = [d for d in detectors if d.answer == "yes"]
        if attacks:
            max_conf = max(d.percentageOfConfidence for d in attacks)
            return FilterResult(answer="yes", percentageOfConfidence=round(max_conf, 1))
        min_conf = min(d.percentageOfConfidence for d in detectors)
        return FilterResult(answer="no", percentageOfConfidence=round(min_conf, 1))

    @staticmethod
    def _aggregate_verdict(filters: VerifyFiltersResponse, threshold: float) -> Verdict:
        """Combina los filtros con pesos para producir el veredicto final."""
        w = VERDICT_WEIGHTS

        def _not_attack_score(result: FilterResult) -> float:
            conf = result.percentageOfConfidence / 100.0
            return conf if result.answer == "no" else 1.0 - conf

        scores = {
            "screen":  _not_attack_score(filters.screen_capture),
            "printed": _not_attack_score(filters.printed_paper),
            "forged":  _not_attack_score(filters.superimposed_elements),
            "ai":      _not_attack_score(filters.ai_altered),
            "live":    _not_attack_score(filters.liveness),
        }
        overall = (
            w.screen_capture * scores["screen"]
            + w.printed_paper * scores["printed"]
            + w.superimposed_elements * scores["forged"]
            + w.ai_altered * scores["ai"]
            + w.liveness * scores["live"]
        )
        overall = max(0.0, min(1.0, overall))

        any_attack = (
            (filters.screen_capture.answer == "yes" and filters.screen_capture.percentageOfConfidence > 75)
            or (filters.printed_paper.answer == "yes" and filters.printed_paper.percentageOfConfidence > 75)
            or (filters.superimposed_elements.answer == "yes" and filters.superimposed_elements.percentageOfConfidence > 65)
            or (filters.ai_altered.answer == "yes" and filters.ai_altered.percentageOfConfidence > 60)
        )

        is_authentic = not any_attack and overall >= threshold

        if overall >= 0.8:
            risk = RiskLevel.low
        elif overall >= 0.6:
            risk = RiskLevel.medium
        else:
            risk = RiskLevel.high

        return Verdict(
            is_authentic=is_authentic,
            overall_confidence=round(overall * 100, 1),
            risk_level=risk,
        )
