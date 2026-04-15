"""
Tests completos para la API de verificación de documentos.
Valida contrato JSON, pipeline Gemini-first con fallback local, y edge cases.

Sin GEMINI_API_KEY en env, el pipeline usa modelos locales automáticamente.
"""

import base64
import io
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from app.schemas import FilterResult, VerifyFiltersResponse, VerifyRequest, VerifyResponse


def make_test_image(width: int = 256, height: int = 160, color: tuple = (140, 150, 170)) -> str:
    """Genera imagen de prueba en base64."""
    image = Image.new("RGB", (width, height), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def make_gradient_image(width: int = 256, height: int = 160) -> str:
    """Genera imagen con gradiente para tests más realistas."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        arr[i, :, 0] = int(255 * i / height)
        arr[i, :, 1] = 128
        arr[i, :, 2] = int(255 * (1 - i / height))
    image = Image.fromarray(arr)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ── Schema tests ───────────────────────────────────────────────
class TestSchemas:
    def test_filter_result_valid(self) -> None:
        r = FilterResult(answer="yes", percentageOfConfidence=95.5)
        assert r.answer == "yes"
        assert r.percentageOfConfidence == 95.5

    def test_filter_result_bounds(self) -> None:
        with pytest.raises(Exception):
            FilterResult(answer="yes", percentageOfConfidence=101)
        with pytest.raises(Exception):
            FilterResult(answer="yes", percentageOfConfidence=-1)

    def test_verify_request_min_image(self) -> None:
        with pytest.raises(Exception):
            VerifyRequest(image="short")

    def test_verify_request_defaults(self) -> None:
        req = VerifyRequest(image="a" * 100)
        assert req.options.confidence_threshold == 0.5
        assert req.options.return_heatmaps is False


# ── Gemini engine tests ────────────────────────────────────────
class TestGeminiEngine:
    def test_engine_disabled_without_api_key(self) -> None:
        from app.filters.gemini_engine import GeminiEngine

        engine = GeminiEngine()
        with patch.dict("os.environ", {}, clear=True):
            engine.initialize()
        assert engine.available is False

    def test_parse_response_valid_json(self) -> None:
        from app.filters.gemini_engine import _parse_gemini_response

        result = _parse_gemini_response('{"detected": true, "confidence": 85.5, "reason": "moiré"}')
        assert result is not None
        assert result["detected"] is True
        assert result["confidence"] == 85.5

    def test_parse_response_wrapped_markdown(self) -> None:
        from app.filters.gemini_engine import _parse_gemini_response

        text = '```json\n{"detected": false, "confidence": 90, "reason": "clean"}\n```'
        result = _parse_gemini_response(text)
        assert result is not None
        assert result["detected"] is False

    def test_parse_response_garbage(self) -> None:
        from app.filters.gemini_engine import _parse_gemini_response

        result = _parse_gemini_response("This is not JSON at all")
        assert result is None

    def test_analyze_filter_returns_none_when_disabled(self) -> None:
        from app.filters.gemini_engine import GeminiEngine

        engine = GeminiEngine()
        # Sin inicializar → no disponible
        img = Image.new("RGB", (100, 100))
        result = engine.analyze_filter(img, "screen_capture")
        assert result is None


# ── Filter unit tests (fallback local) ─────────────────────────
class TestScreenCaptureFilter:
    def test_predict_returns_filter_result(self) -> None:
        from app.filters.screen_capture import ScreenCaptureDetector

        detector = ScreenCaptureDetector()
        img = Image.new("RGB", (224, 224), color=(100, 120, 140))
        result = detector.predict(img)
        assert isinstance(result, FilterResult)
        assert result.answer in ("yes", "no")
        assert 0 <= result.percentageOfConfidence <= 100


class TestPrintedPaperFilter:
    def test_predict_returns_filter_result(self) -> None:
        from app.filters.printed_paper import PrintedPaperDetector

        detector = PrintedPaperDetector()
        img = Image.new("RGB", (380, 380), color=(200, 200, 200))
        result = detector.predict(img)
        assert isinstance(result, FilterResult)
        assert result.answer in ("yes", "no")
        assert 0 <= result.percentageOfConfidence <= 100


class TestForgeryDetector:
    def test_predict_returns_filter_result(self) -> None:
        from app.filters.forgery_detection import ForgeryDetector

        detector = ForgeryDetector()
        img = Image.new("RGB", (256, 256), color=(180, 180, 180))
        result = detector.predict(img)
        assert isinstance(result, FilterResult)
        assert result.answer in ("yes", "no")
        assert 0 <= result.percentageOfConfidence <= 100


class TestAIDetector:
    def test_predict_returns_filter_result(self) -> None:
        from app.filters.ai_detection import AIAlteredDetector

        detector = AIAlteredDetector()
        img = Image.new("RGB", (224, 224), color=(150, 150, 150))
        result = detector.predict(img)
        assert isinstance(result, FilterResult)
        assert result.answer in ("yes", "no")
        assert 0 <= result.percentageOfConfidence <= 100


class TestLivenessDetector:
    def test_predict_returns_filter_result(self) -> None:
        from app.filters.liveness import LivenessDetector

        detector = LivenessDetector()
        img = Image.new("RGB", (256, 256), color=(170, 170, 170))
        screen = FilterResult(answer="no", percentageOfConfidence=90.0)
        printed = FilterResult(answer="no", percentageOfConfidence=85.0)
        result = detector.predict(img, screen, printed)
        assert isinstance(result, FilterResult)
        assert result.answer in ("yes", "no")
        assert 0 <= result.percentageOfConfidence <= 100


# ── Pipeline tests ─────────────────────────────────────────────
class TestPipeline:
    def test_pipeline_falls_back_to_local_without_gemini(self) -> None:
        """Sin GEMINI_API_KEY, el pipeline debe usar modelos locales."""
        from app.pipeline import VerificationPipeline

        pipeline = VerificationPipeline()
        with patch.dict("os.environ", {}, clear=True):
            pipeline.load_models()
        assert pipeline.gemini.available is False

    def test_decode_image_valid(self) -> None:
        from app.pipeline import VerificationPipeline

        b64 = make_test_image()
        img = VerificationPipeline._decode_image(b64)
        assert img.mode == "RGB"

    def test_decode_image_oversized(self) -> None:
        from app.pipeline import VerificationPipeline

        img = Image.new("RGB", (5000, 5000))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        with pytest.raises(ValueError, match="4096"):
            VerificationPipeline._decode_image(b64)


# ── API integration tests ──────────────────────────────────────
class TestAPI:
    @pytest.fixture(autouse=True)
    def setup_client(self) -> None:
        from fastapi.testclient import TestClient

        from app.main import app

        self.client = TestClient(app)

    def test_health(self) -> None:
        resp = self.client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["engine"] in ("gemini", "local")

    def test_verify_full_contract(self) -> None:
        payload = {
            "image": make_test_image(),
            "options": {"return_heatmaps": False, "confidence_threshold": 0.5},
        }
        resp = self.client.post("/verify", json=payload)
        assert resp.status_code == 200

        data = resp.json()
        assert data["status"] == "completed"
        assert "request_id" in data
        assert "processing_time_ms" in data
        assert data["processing_time_ms"] >= 0

        # Verdict
        verdict = data["verdict"]
        assert isinstance(verdict["is_authentic"], bool)
        assert 0 <= verdict["overall_confidence"] <= 100
        assert verdict["risk_level"] in ("low", "medium", "high")

        # Todos los 5 filtros presentes
        filters = data["filters"]
        expected_filters = {"screen_capture", "printed_paper", "superimposed_elements", "ai_altered", "liveness"}
        assert set(filters.keys()) == expected_filters

        for name in expected_filters:
            assert filters[name]["answer"] in ("yes", "no")
            assert 0 <= filters[name]["percentageOfConfidence"] <= 100

    def test_verify_with_gradient_image(self) -> None:
        payload = {"image": make_gradient_image()}
        resp = self.client.post("/verify", json=payload)
        assert resp.status_code == 200
        assert set(resp.json()["filters"].keys()) == {
            "screen_capture", "printed_paper", "superimposed_elements", "ai_altered", "liveness",
        }

    def test_verify_invalid_base64(self) -> None:
        payload = {"image": "not_valid_base64_" * 10}
        resp = self.client.post("/verify", json=payload)
        assert resp.status_code == 400

    def test_verify_oversized_image(self) -> None:
        img = Image.new("RGB", (5000, 5000), color=(0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        payload = {"image": b64}
        resp = self.client.post("/verify", json=payload)
        assert resp.status_code == 400
        assert "4096" in resp.json()["detail"]
