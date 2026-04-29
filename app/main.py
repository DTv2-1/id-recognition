"""
ADAMO ID — FastAPI endpoint principal.
Motor primario: Gemini Vision API.
Fallback: modelos locales (PyTorch/ONNX).
"""

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()  # carga .env desde la raíz del proyecto

from fastapi import FastAPI, HTTPException

from app.pipeline import VerificationPipeline
from app.schemas import VerifyRequest, VerifyResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

pipeline = VerificationPipeline()


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Inicializa Gemini y/o modelos locales."""
    pipeline.load_models()
    yield


app = FastAPI(
    title="ADAMO ID Verification API",
    version="0.2.0",
    description=(
        "API de verificación de autenticidad de documentos de identidad con 5 filtros de IA. "
        "Motor primario: Gemini Vision. Fallback: modelos locales."
    ),
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "engine": "gemini" if pipeline.gemini.available else "local",
    }


@app.post("/verify", response_model=VerifyResponse)
def verify(payload: VerifyRequest) -> VerifyResponse:
    try:
        return pipeline.verify(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc
