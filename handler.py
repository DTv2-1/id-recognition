"""
ADAMO ID — RunPod Serverless Handler.
Entry point para deployment en RunPod o como servidor standalone.

Motor primario: Gemini Vision API (GEMINI_API_KEY env var).
Fallback: modelos locales PyTorch/ONNX.
"""

import logging
import os

from app.pipeline import VerificationPipeline
from app.schemas import VerifyRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Carga global (una vez por cold start) ──────────────────────
pipeline = VerificationPipeline()
device = os.environ.get("DEVICE", "cpu")
pipeline.load_models(device=device)
engine = "gemini" if pipeline.gemini.available else "local"
logger.info("Pipeline ready — engine=%s device=%s", engine, device)


def handler(job: dict) -> dict:
    """Handler: recibe job con input, retorna resultado JSON."""
    try:
        payload = VerifyRequest(**job.get("input", {}))
        result = pipeline.verify(payload)
        return result.model_dump(mode="json")
    except ValueError as exc:
        return {"error": str(exc), "status": "failed"}
    except Exception as exc:
        logger.exception("Unexpected error in handler")
        return {"error": f"Internal error: {exc}", "status": "failed"}


if __name__ == "__main__":
    # Si hay runpod instalado, usar serverless. Si no, FastAPI.
    try:
        import runpod
        runpod.serverless.start({"handler": handler})
    except ImportError:
        import uvicorn
        from app.main import app
        uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
