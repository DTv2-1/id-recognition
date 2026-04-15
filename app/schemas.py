"""
ADAMO ID — Modelos Pydantic para request/response de la API.
Contrato JSON completo con los 5 filtros de verificación.
"""

from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class BinaryAnswer(str, Enum):
    yes = "yes"
    no = "no"


class FilterResult(BaseModel):
    answer: BinaryAnswer
    percentageOfConfidence: float = Field(ge=0, le=100)


class VerifyOptions(BaseModel):
    return_heatmaps: bool = False
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class VerifyRequest(BaseModel):
    image: str = Field(min_length=32)
    options: VerifyOptions = VerifyOptions()


class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class Verdict(BaseModel):
    is_authentic: bool
    overall_confidence: float = Field(ge=0, le=100)
    risk_level: RiskLevel


class VerifyFiltersResponse(BaseModel):
    screen_capture: FilterResult
    printed_paper: FilterResult
    superimposed_elements: FilterResult
    ai_altered: FilterResult
    liveness: FilterResult


class VerifyResponse(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    request_id: UUID
    status: str = "completed"
    processing_time_ms: int = Field(ge=0)
    verdict: Verdict
    filters: VerifyFiltersResponse
