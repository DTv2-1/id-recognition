"""
Motor de verificación basado en Gemini Vision API.

Una sola llamada por imagen con un prompt unificado. Gemini devuelve un
veredicto GENUINE | FRAUD + un fraud_type, y eso se descompone en los
5 FilterResult que el resto del pipeline consume.

Mucho más rápido (~5 s/imagen vs 30 s+ del esquema antiguo de 5 llamadas)
y más coherente, porque un solo modelo razona sobre toda la evidencia.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from app.filters.forensic_features import compute_features
from app.schemas import FilterResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Prompt unificado: una sola llamada decide entre 5 categorías de fraude
# o GENUINE.
# ─────────────────────────────────────────────────────────────────────
_PROMPT_UNIFIED = """You are a senior forensic document examiner trained to spot PAPER PRINTOUTS of Colombian cédulas. The most common fraud you face is someone printing a scanned ID on a regular sheet of paper, cutting it out, and re-photographing it. You must NOT assume "looks like a card → is a card". You must actively verify physicality.

═══════════════════════════════════════════════════════════════════════
STEP 1 — MANDATORY POSITIVE-EVIDENCE CHECKLIST FOR REAL POLYCARBONATE
═══════════════════════════════════════════════════════════════════════
For each item, decide YES / NO / UNCLEAR. A real plastic cédula will satisfy
several of these. A paper printout will satisfy NONE or maybe one by accident.

  1. LATERAL THICKNESS: can you see a small white/cream lateral edge ~0.8mm
     thick on at least one side of the card? (Real polycarbonate has a
     visible side wall; paper printouts are razor-flat.)
  2. PORTRAIT ENGRAVING: does the portrait have the characteristic black-and-
     gray laser-engraved look with horizontal raster lines, sharp embedded
     into the substrate? (NOT a soft pinkish/colored portrait that looks
     printed on top.)
  3. GHOST IMAGE / SECONDARY PORTRAIT: is there a visible smaller secondary
     portrait window with its own engraving? (Hard to fake on print.)
  4. CRISP MICROPRINT & GUILLOCHE: are the background guilloche curves and
     the fine red/yellow security lines sharp, vivid, with strong contrast
     and no halftone dot pattern when zoomed?
  5. RAISED / TACTILE TEXT: any indication of relief on the cédula number
     or signature (slight shadow on one side suggesting embossing)?
  6. ROUNDED CORNER PROFILE: do the corners show smooth machine-cut
     polycarbonate radius (NOT a slightly fuzzy / fibrous paper-cut edge)?

═══════════════════════════════════════════════════════════════════════
STEP 2 — SCAN FOR FRAUD SIGNALS
═══════════════════════════════════════════════════════════════════════

  A) SCREEN_CAPTURE — visible mouse cursor / I-beam, device bezels or status
     bars, moiré interference, pixel grid, self-illuminated card brighter
     than its surroundings, BLACK BEZEL / FRAME around the card. Cursor
     anywhere → FRAUD ≥95.

  B) PRINTED_PAPER — paper printout / photocopy. Strong signs:
     • RAZOR-FLAT card with NO visible lateral edge anywhere.
     • Colors WASHED, PINK / MAGENTA tinted, or YELLOWED.
     • Portrait looks SOFT, PINKISH, BLENDED into the background instead of
       sharply laser-engraved (no horizontal raster lines visible).
     • Card sitting on TEXTILE / FABRIC / BEDDING / RUFFLED PAPER /
       NOTEBOOK PAPER WITH HORIZONTAL OR GRID LINES.
     • Signature looks DRAWN ON TOP with pen/marker (different ink texture
       or gloss than the rest of the print).
     • Visible fold / crease / paper-cut edge, paper fibers, halftone CMYK
       dots, inkjet banding, color-registration mis-alignment, or full sheet
       of A4 paper visible around the card.
     • Background guilloche looks FAINT, BLURRY or DESATURATED — typical of
       the second-generation copy that a printout produces.

  C) SUPERIMPOSED — stickers, white opaque patches over data fields,
     glued/swapped portrait, rectangular cut-edges around the photo,
     security pattern stopping at the portrait/sticker border.

  D) AI_GENERATED — melting facial features, dissolved hair edges, glowing
     unnatural eye color, fonts that DO NOT match the official ID typography
     (wrong serif/weight in NUIP block).
     ⚠️ Do NOT flag AI just because the portrait looks young, smooth, has
     warm eye color, or any single subjective "weird" feeling. Require
     concrete artefacts (asymmetric eyes, broken teeth, gibberish text).

  E) DATA_INCONSISTENCY (treat as SUPERIMPOSED — the data was tampered):
     • Card design vs expedition date mismatch (modern design with very old
       date, or vice-versa).
     • Registrador name vs date mismatch — example: "ALEXANDER VEGA ROCHA"
       only signed from ~2015 onward, so any earlier date with him = fake.
     • NUIP that is obviously fake: sequential like "1.001.234.661",
       round number "1.000.000.000", repeated digits.
     Any clear inconsistency alone → FRAUD ≥90, fraud_type=SUPERIMPOSED.

═════════════════════════════════════════════════════════════════════
STEP 3 — DECISION RULE  (READ CAREFULLY)
═════════════════════════════════════════════════════════════════════

Note: if the FORENSIC PRE-ANALYSIS block contains a ⚠ alert line
(LOW SAT + NO SPECULAR), treat it as strong supporting evidence for
PRINTED_PAPER, but still require at least one visual signal from STEP 2B.

Count how many of the 6 STEP-1 checks returned YES (positive evidence of
real polycarbonate).

  • 0 YES   → suspicious, but NOT automatic FRAUD. Only flag PRINTED_PAPER
              if you ALSO see at least one STEP-2B visual signal (pink cast,
              washed guilloche, textile background, visible paper fiber, etc.)
              OR the FORENSIC ⚠ alert is present.
              Many genuine card photos fail all STEP-1 checks due to camera
              angle, low resolution, or indoor lighting — that alone is NOT
              evidence of paper.
  • 1 YES   → likely genuine. Only flag PRINTED_PAPER if there is a CLEAR
              and SPECIFIC visual signal from STEP-2B.
  • 2+ YES  → real card. Verdict GENUINE unless STEP-2 catches a specific
              fraud signal (cursor, sticker, anachronism…).

⚠️ ANTI-FALSE-POSITIVE GUARDS (to keep authentic photos passing):
  • A real card on a wooden table, desk, hand, or plain paper background
    is NORMAL. Only treat the surface as a clue if it is clearly textile/
    bedding/lined notebook.
  • Holograms are often invisible at the camera angle — absence of visible
    hologram alone is NOT fraud.
  • Slight blur, JPEG compression, warm indoor lighting → NOT fraud.
  • Colored official banners ("REPÚBLICA DE COLOMBIA", blue field labels)
    are PART of the design — never flag as stickers.
  • If the image is dark/blurry but you can still see a visible 0.8mm
    lateral edge OR clear laser-engraved portrait → GENUINE.
  • Do NOT invoke AI_GENERATED for "young face / warm eyes / smooth skin".

═══════════════════════════════════════════════════════════════════════
OUTPUT FORMAT — respond ONLY with this exact JSON (no markdown, no prose):
═══════════════════════════════════════════════════════════════════════
{
  "checks": {
    "lateral_thickness":   "yes" | "no" | "unclear",
    "portrait_engraving":  "yes" | "no" | "unclear",
    "ghost_image":         "yes" | "no" | "unclear",
    "crisp_guilloche":     "yes" | "no" | "unclear",
    "raised_text":         "yes" | "no" | "unclear",
    "rounded_corners":     "yes" | "no" | "unclear"
  },
  "yes_count": <integer 0-6>,
  "verdict": "GENUINE" | "FRAUD",
  "confidence": 0-100,
  "fraud_type": null | "SCREEN_CAPTURE" | "PRINTED_PAPER" | "SUPERIMPOSED" | "AI_GENERATED",
  "reason": "one or two sentences citing the specific visual evidence"
}
"""


# Mapeo fraud_type → nombre del filtro interno usado por el pipeline.
# NOTA: "liveness" NO aparece aquí: es un score consolidado de los 4 filtros
# de fraude, calculado en pipeline.py (es "el filtro 5" — grado de
# autenticidad global, NO un detector independiente).
_FRAUD_TYPE_TO_FILTER: dict[str, str] = {
    "SCREEN_CAPTURE": "screen_capture",
    "PRINTED_PAPER": "printed_paper",
    "SUPERIMPOSED": "superimposed_elements",
    "AI_GENERATED": "ai_altered",
}

# Nombres de los 4 filtros de fraude que Gemini evalúa directamente.
# El 5º ("liveness") se deriva en el pipeline.
FILTER_NAMES: tuple[str, ...] = (
    "screen_capture",
    "printed_paper",
    "superimposed_elements",
    "ai_altered",
)


# ─────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────
def _build_gemini_client():
    """Crea el cliente de Gemini. Requiere GEMINI_API_KEY en env."""
    try:
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set — Gemini engine disabled.")
            return None
        # NOTA: http_options.timeout está en MILISEGUNDOS en google-genai SDK
        # api_version='v1alpha' habilita media_resolution para Gemini 3.
        return genai.Client(
            api_key=api_key,
            http_options={"timeout": 180_000, "api_version": "v1alpha"},
        )
    except ImportError:
        logger.warning("google-genai not installed — Gemini engine disabled.")
        return None


def _image_to_bytes(image: Image.Image, max_px: int = 1600) -> bytes:
    """PIL Image → JPEG bytes, redimensionando para acotar tokens de entrada."""
    img = image.copy()
    w, h = img.size
    if max(w, h) > max_px:
        ratio = max_px / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────
# Motor
# ─────────────────────────────────────────────────────────────────────
@dataclass
class GeminiEngine:
    """Motor de verificación usando Gemini Vision API — 1 llamada unificada."""

    model_name: str = "gemini-3-flash-preview"
    _client: object | None = field(default=None, repr=False)
    _available: bool = field(default=False, repr=False)

    def initialize(self) -> None:
        self._client = _build_gemini_client()
        self._available = self._client is not None
        if self._available:
            logger.info("Gemini engine initialized with model=%s", self.model_name)
        else:
            logger.warning("Gemini engine NOT available — pipeline will use local fallback.")

    @property
    def available(self) -> bool:
        return self._available

    # ── API pública usada por el pipeline ──────────────────────────────
    def analyze_all(self, image: Image.Image) -> dict[str, FilterResult | None]:
        """Llama a Gemini una sola vez y devuelve dict {filter_name: FilterResult}."""
        if not self._available or self._client is None:
            return {name: None for name in FILTER_NAMES}
        return self._analyze_unified(image)

    # ── Implementación ─────────────────────────────────────────────────
    def _analyze_unified(self, image: Image.Image) -> dict[str, FilterResult | None]:
        from google.genai import types

        img_bytes = _image_to_bytes(image)

        # Pre-análisis forense local (numpy/PIL) — se inyecta como contexto
        # objetivo en el prompt para romper la "physical blindness" del VLM.
        try:
            features = compute_features(image)
            forensic_block = features.as_prompt_block()
            forensic_alert = (
                features.color_saturation_p95 < 0.30
                and features.specular_count <= 1
            )
        except Exception as exc:  # pragma: no cover — defensivo
            logger.warning("forensic_features failed: %s", exc)
            forensic_block = ""
            forensic_alert = False

        full_prompt = (
            forensic_block + "\n" + _PROMPT_UNIFIED if forensic_block else _PROMPT_UNIFIED
        )

        try:
            print("  → Gemini [unified] enviando...", flush=True)
            t0 = time.time()
            response = self._client.models.generate_content(  # type: ignore[union-attr]
                model=self.model_name,
                contents=[types.Content(role="user", parts=[
                    types.Part(text=full_prompt),
                    types.Part(
                        inline_data=types.Blob(mime_type="image/jpeg", data=img_bytes),
                        media_resolution={"level": "media_resolution_high"},
                    ),
                ])],
                config=types.GenerateContentConfig(
                    max_output_tokens=4096,
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_level="high"),
                ),
            )
            data = json.loads(response.text or "{}")
        except Exception as exc:
            logger.warning("Gemini [unified] failed: %s", exc)
            print(f"  ✗ [unified] ERROR: {exc}", flush=True)
            return {name: None for name in FILTER_NAMES}

        verdict = str(data.get("verdict", "GENUINE")).upper()
        confidence = max(0.0, min(100.0, float(data.get("confidence", 50.0))))
        fraud_type = data.get("fraud_type")
        reason = data.get("reason", "")

        # OVERRIDE forense: si la alerta de printout se disparó pero Gemini
        # decidió GENUINE, ignoramos su veredicto. La regla
        # (saturation_p95 < 0.30 AND specular ≤ 1) es muy específica y casi
        # nunca se cumple en cédulas auténticas (validado offline).
        if forensic_alert and verdict == "GENUINE":
            logger.info(
                "Forensic OVERRIDE: GENUINE → FRAUD/PRINTED_PAPER "
                "(sat_p95=%.2f, spec=%d)",
                features.color_saturation_p95, features.specular_count,
            )
            print(
                f"  ⚡ OVERRIDE forensic: GENUINE → FRAUD/PRINTED_PAPER "
                f"(sat={features.color_saturation_p95:.2f}, spec={features.specular_count})",
                flush=True,
            )
            verdict = "FRAUD"
            fraud_type = "PRINTED_PAPER"
            confidence = max(confidence, 85.0)
            reason = f"[forensic-override] {reason}"

        print(
            f"  ✓ [unified] {verdict} {confidence:.0f}% "
            f"({fraud_type or '-'}, {time.time() - t0:.1f}s)",
            flush=True,
        )
        if reason:
            logger.info(
                "Gemini [unified] %s/%s conf=%.1f — %s",
                verdict, fraud_type, confidence, reason,
            )

        flagged_filter = (
            _FRAUD_TYPE_TO_FILTER.get(str(fraud_type).upper()) if fraud_type else None
        )
        # Si Gemini dice FRAUD pero no especifica tipo → marcamos superimposed
        # (el "cajón de sastre" más frecuente en frauds reales).
        if verdict == "FRAUD" and flagged_filter is None:
            flagged_filter = "superimposed_elements"

        results: dict[str, FilterResult | None] = {}
        for name in FILTER_NAMES:
            if verdict == "FRAUD" and name == flagged_filter:
                results[name] = FilterResult(
                    answer="yes", percentageOfConfidence=round(confidence, 1)
                )
            else:
                # Para los demás filtros: "no" con la confianza global.
                # En GENUINE usamos la confianza directa (suele ser 85-95).
                # En FRAUD elevamos a ≥85 para no contaminar el agregado.
                clean_conf = confidence if verdict == "GENUINE" else max(85.0, confidence)
                results[name] = FilterResult(
                    answer="no", percentageOfConfidence=round(clean_conf, 1)
                )
        return results
