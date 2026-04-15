"""
ADAMO ID — Pipeline de verificación de documentos.
Estrategia: Gemini Vision como motor primario, modelos locales como fallback.
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
    VerifyFiltersResponse,
    VerifyRequest,
    VerifyResponse,
    Verdict,
)

logger = logging.getLogger(__name__)


def _load_local_filters():
    """Carga lazy de filtros locales (pesados) solo cuando se necesiten."""
    from app.filters.ai_detection import AIAlteredDetector
    from app.filters.forgery_detection import ForgeryDetector
    from app.filters.liveness import LivenessDetector
    from app.filters.printed_paper import PrintedPaperDetector
    from app.filters.screen_capture import ScreenCaptureDetector

    return {
        "screen_capture": ScreenCaptureDetector(),
        "printed_paper": PrintedPaperDetector(),
        "superimposed_elements": ForgeryDetector(),
        "ai_altered": AIAlteredDetector(),
        "liveness": LivenessDetector(),
    }


@dataclass
class VerificationPipeline:
    """Pipeline con Gemini como primario y modelos locales como fallback."""

    gemini: GeminiEngine = field(default_factory=GeminiEngine)
    _local_filters: dict | None = field(default=None, repr=False)
    _local_loaded: bool = field(default=False, repr=False)

    def load_models(self, device: str | None = None) -> None:
        """
        Inicializa el pipeline.
        - Siempre intenta conectar con Gemini.
        - Solo pre-carga modelos locales si Gemini no está disponible.
        - Si Gemini está activo, modelos locales se cargan on-demand (lazy).
        """
        self.gemini.initialize()

        if not self.gemini.available:
            logger.info("Gemini not available — pre-loading local models as primary.")
            self._ensure_local_filters(device)
        else:
            logger.info("Gemini available — local models will load on-demand if needed.")

    def _ensure_local_filters(self, device: str | None = None) -> None:
        """Carga modelos locales si aún no están cargados."""
        if self._local_loaded:
            return

        try:
            import torch
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        logger.info("Loading local fallback models on device=%s", device)
        t0 = time.perf_counter()
        self._local_filters = _load_local_filters()

        for name, detector in self._local_filters.items():
            try:
                detector.load_model(device)
            except Exception as exc:
                logger.warning("Failed to load local filter '%s': %s", name, exc)

        self._local_loaded = True
        elapsed = time.perf_counter() - t0
        logger.info("Local models loaded in %.1fs", elapsed)

    def verify(self, payload: VerifyRequest) -> VerifyResponse:
        """Ejecuta los 5 filtros y produce el veredicto."""
        start = time.perf_counter()
        image = self._decode_image(payload.image)

        if self.gemini.available:
            result = self._verify_with_gemini(image, payload)
        else:
            result = self._verify_with_local(image, payload)

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        result.processing_time_ms = elapsed_ms
        return result

    def _verify_with_gemini(self, image: Image.Image, payload: VerifyRequest) -> VerifyResponse:
        """Ejecuta verificación con Gemini, con fallback local por filtro."""
        gemini_results = self.gemini.analyze_all(image)

        screen_capture = gemini_results.get("screen_capture")
        printed_paper = gemini_results.get("printed_paper")
        superimposed = gemini_results.get("superimposed_elements")
        ai_altered = gemini_results.get("ai_altered")
        liveness = gemini_results.get("liveness")

        # Fallback local solo donde Gemini falló
        needs_local = any(r is None for r in [screen_capture, printed_paper, superimposed, ai_altered, liveness])
        if needs_local:
            self._ensure_local_filters()

        if screen_capture is None and self._local_filters:
            logger.info("Fallback local: screen_capture")
            screen_capture = self._local_filters["screen_capture"].predict(image)

        if printed_paper is None and self._local_filters:
            logger.info("Fallback local: printed_paper")
            printed_paper = self._local_filters["printed_paper"].predict(image)

        if superimposed is None and self._local_filters:
            logger.info("Fallback local: superimposed_elements")
            superimposed = self._local_filters["superimposed_elements"].predict(image)

        if ai_altered is None and self._local_filters:
            logger.info("Fallback local: ai_altered")
            ai_altered = self._local_filters["ai_altered"].predict(image)

        if liveness is None and self._local_filters:
            logger.info("Fallback local: liveness")
            liveness = self._local_filters["liveness"].predict(image, screen_capture, printed_paper)

        # Si después de todo aún hay None, usar resultado neutro
        if screen_capture is None:
            screen_capture = FilterResult(answer="no", percentageOfConfidence=50.0)
        if printed_paper is None:
            printed_paper = FilterResult(answer="no", percentageOfConfidence=50.0)
        if superimposed is None:
            superimposed = FilterResult(answer="no", percentageOfConfidence=50.0)
        if ai_altered is None:
            ai_altered = FilterResult(answer="no", percentageOfConfidence=50.0)
        if liveness is None:
            liveness = FilterResult(answer="yes", percentageOfConfidence=50.0)

        filters = VerifyFiltersResponse(
            screen_capture=screen_capture,
            printed_paper=printed_paper,
            superimposed_elements=superimposed,
            ai_altered=ai_altered,
            liveness=liveness,
        )
        verdict = self._aggregate_verdict(filters, payload.options.confidence_threshold)
        request_id = uuid4()

        logger.info(
            "request_id=%s [GEMINI] verdict=%s confidence=%.1f risk=%s",
            request_id, verdict.is_authentic, verdict.overall_confidence, verdict.risk_level,
        )

        return VerifyResponse(
            request_id=request_id,
            processing_time_ms=0,
            verdict=verdict,
            filters=filters,
        )

    def _verify_with_local(self, image: Image.Image, payload: VerifyRequest) -> VerifyResponse:
        """Ejecuta verificación con modelos locales (fallback completo)."""
        self._ensure_local_filters()

        screen_capture = self._local_filters["screen_capture"].predict(image)
        printed_paper = self._local_filters["printed_paper"].predict(image)
        superimposed = self._local_filters["superimposed_elements"].predict(image)
        ai_altered = self._local_filters["ai_altered"].predict(image)
        liveness = self._local_filters["liveness"].predict(image, screen_capture, printed_paper)

        filters = VerifyFiltersResponse(
            screen_capture=screen_capture,
            printed_paper=printed_paper,
            superimposed_elements=superimposed,
            ai_altered=ai_altered,
            liveness=liveness,
        )
        verdict = self._aggregate_verdict(filters, payload.options.confidence_threshold)
        request_id = uuid4()

        logger.info(
            "request_id=%s [LOCAL] verdict=%s confidence=%.1f risk=%s",
            request_id, verdict.is_authentic, verdict.overall_confidence, verdict.risk_level,
        )

        return VerifyResponse(
            request_id=request_id,
            processing_time_ms=0,
            verdict=verdict,
            filters=filters,
        )

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
    def _aggregate_verdict(filters: VerifyFiltersResponse, threshold: float) -> Verdict:
        """
        Combina los 5 filtros con pesos configurables para el veredicto final.
        """
        w = VERDICT_WEIGHTS

        def _not_attack_score(result: FilterResult, invert: bool = False) -> float:
            conf = result.percentageOfConfidence / 100.0
            if invert:
                return conf if result.answer == "yes" else 1.0 - conf
            else:
                return conf if result.answer == "no" else 1.0 - conf

        scores = {
            "screen": _not_attack_score(filters.screen_capture),
            "printed": _not_attack_score(filters.printed_paper),
            "forged": _not_attack_score(filters.superimposed_elements),
            "ai": _not_attack_score(filters.ai_altered),
            "live": _not_attack_score(filters.liveness, invert=True),
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
            (filters.screen_capture.answer == "yes" and filters.screen_capture.percentageOfConfidence > 60)
            or (filters.printed_paper.answer == "yes" and filters.printed_paper.percentageOfConfidence > 60)
            or (filters.superimposed_elements.answer == "yes" and filters.superimposed_elements.percentageOfConfidence > 60)
            or (filters.ai_altered.answer == "yes" and filters.ai_altered.percentageOfConfidence > 60)
            or (filters.liveness.answer == "no" and filters.liveness.percentageOfConfidence > 60)
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
