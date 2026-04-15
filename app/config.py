"""
ADAMO ID — Configuración central del pipeline de verificación.
Pesos de filtros, umbrales, rutas de modelos y constantes.
"""

from dataclasses import dataclass, field
from pathlib import Path

# ── Rutas base ──────────────────────────────────────────────────
WEIGHTS_DIR = Path("/app/weights")  # En Docker
LOCAL_WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"  # Dev local


def weights_dir() -> Path:
    """Retorna la carpeta de pesos que exista (Docker o local)."""
    if WEIGHTS_DIR.exists():
        return WEIGHTS_DIR
    LOCAL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    return LOCAL_WEIGHTS_DIR


# ── Validación de inputs ────────────────────────────────────────
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_DIMENSION = 4096
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

# ── Filtro 1: Screen Capture (CMA ViT-B/16) ────────────────────
SCREEN_MODEL_INPUT_SIZE = 224
SCREEN_IMAGENET_MEAN = [0.485, 0.456, 0.406]
SCREEN_IMAGENET_STD = [0.229, 0.224, 0.225]

# ── Filtro 2: Print Detection (EfficientNet-B4) ────────────────
PRINT_MODEL_INPUT_SIZE = 380  # Patch size recomendado para halftone
PRINT_FFT_BAND_LOW = 30      # Frecuencia baja de banda de interés
PRINT_FFT_BAND_HIGH = 120    # Frecuencia alta de banda de interés

# ── Filtro 3: Forgery Detection (HiFi-IFDL) ────────────────────
FORGERY_MODEL_INPUT_SIZE = 256
FORGERY_THRESHOLD = 0.5

# ── Filtro 4: AI Detection (UnivFD / CLIP ViT-L/14) ────────────
AI_MODEL_INPUT_SIZE = 224
AI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
AI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
AI_AMBIGUOUS_LOW = 0.30   # Si UnivFD da entre 30-70%, activar DIRE
AI_AMBIGUOUS_HIGH = 0.70

# ── Filtro 5: Liveness Composite ───────────────────────────────
FACE_ANTISPOOF_INPUT_SIZE = (80, 80)
FACE_ANTISPOOF_REAL_LABEL = 1  # label==1 → real face

@dataclass
class LivenessWeights:
    """Pesos para el scoring compuesto de liveness."""
    not_screen: float = 0.25
    not_printed: float = 0.20
    face_live: float = 0.20
    perspective_valid: float = 0.15
    hand_detected: float = 0.10
    exif_valid: float = 0.10

LIVENESS_WEIGHTS = LivenessWeights()

# ── Veredicto final ────────────────────────────────────────────
@dataclass
class VerdictWeights:
    """Pesos para agregar los 5 filtros en el veredicto final."""
    screen_capture: float = 0.20
    printed_paper: float = 0.20
    superimposed_elements: float = 0.25
    ai_altered: float = 0.20
    liveness: float = 0.15

VERDICT_WEIGHTS = VerdictWeights()

# ── Gemini API ──────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"  # Mejor relación costo/calidad para visión
GEMINI_FALLBACK_MODEL = "gemini-2.5-flash-lite"  # Más barato, menos capaz
