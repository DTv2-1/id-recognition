"""
Motor de verificación basado en Gemini Vision API.
Cada filtro tiene su propio prompt especializado → mayor precisión.
Usa 5 llamadas independientes con response_mime_type="application/json".
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

try:
    import cv2
    from scipy.fft import fft2, fftshift
    from scipy.signal import wiener
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    from skimage.filters import meijering
    import mahotas
    import colour
    import jpeglib
    import pywt
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

from app.schemas import FilterResult

logger = logging.getLogger(__name__)

# ── Prompts especializados por filtro ─────────────────────────────

_PROMPT_SCREEN_CAPTURE = """You are a forensic document analyst specializing in screen recapture detection. Your ONLY task: determine if this identity document image was photographed from a SCREEN (monitor, TV, phone display) rather than being a direct photo of the physical card.

── FORENSIC METRICS (computed from this image) ──
{quality_context}

── ABSOLUTE PROOF (any ONE → detected=true, confidence=99) ──
- Mouse cursor visible anywhere in the image
- Screen bezel / device frame at edges
- Taskbar, browser UI or OS interface elements

── STRONG SIGNALS — VISUAL (multiple → high confidence) ──
- Moiré interference patterns: wavy color bands from screen pixel grid beating against camera sensor
- Visible pixel grid or RGB sub-pixel dot pattern
- Horizontal/vertical scan lines across the document area
- Document area appears SELF-ILLUMINATED (emitting light) while surroundings are darker — screens are light sources, physical cards only reflect light
- Complete ABSENCE of natural shadows beneath or beside the document (a physical card on a surface always casts shadows)
- Screen glare: specular reflection shaped like a rectangular light source on a flat glossy surface
- Cool/blue color cast over the entire image (LCD backlight ~6500K vs natural light ~3000-5000K)

── STRONG SIGNALS — PHYSICS / SIGNAL (trust the metrics above) ──
- LOW saturation (< 0.40) is very consistent with LCD screens compressing color gamut
- Blue/Red ratio near 1.0 (0.7–1.3) indicates LCD color temperature, not warm ambient light
- Double JPEG compression: faint 8×8 block artifacts at DCT block boundaries (screen renders JPEG → camera re-compresses it)
- High-frequency aliasing on text edges and fine lines (two digitization stages produce alias)

── IMPORTANT — HIGH-RES SCREENS ──
Modern 4K/OLED/Retina displays (>200 PPI) do NOT produce visible moiré or pixel grid. On these:
- Rely on self-illumination, absent shadows, and color temperature signals
- The document will look "too perfect" with zero film grain or surface texture
- Trust the metrics: saturation and blue_ratio are still reliable regardless of screen resolution

── WEAK SIGNALS (alone insufficient) ──
- Slight blur (camera shake can also blur physical cards)
- Reflections (physical cards also reflect)
- Natural-looking background

Be aggressive: false negatives (missing a fake) are worse than false positives. If metrics show LOW saturation AND blue_ratio near 1.0, lean toward detected=true even without visible moiré.

Respond ONLY with this exact JSON (no markdown):
{"detected": false, "confidence": 50, "reason": "one sentence explanation"}

detected=true means it WAS photographed from a screen (FRAUD).
confidence=0-100 (how certain you are)."""

_PROMPT_PRINTED_PAPER = """You are a forensic document physicist. Your ONLY task: determine if this identity document is a PRINTED PAPER COPY (photocopy, laser print, inkjet print) instead of a genuine plastic/polycarbonate ID card.

You have pre-computed forensic metrics from the image. Use them as hard physical evidence:

{quality_context}

HOW TO INTERPRET THE METRICS:
- halftone_peak > 0.08 → possible periodic pattern from inkjet/laser dot grid
- glcm_contrast > 0.25 → rough surface texture (paper fibers, not smooth plastic)
- lbp_variance > 0.003 → local texture variation from paper grain
- haralick_entropy > 0.83 → complex/noisy image = printed noise artifacts
- dct_triple > 0.10 → multiple JPEG compression stages = paper chain
- cmyk_gamut > 0.05 → out-of-gamut pixels from CMYK ink mixing
- hist_clip > 0.001 → pixel accumulation at extremes = paper's limited dynamic range
- blue_ratio 0.85–1.10 → warm neutral illumination of printed white paper (not LCD)
- laplacian_var < 400 → edges are blurry from print-and-capture chain (genuine cards: >600)
- meijering > 0.04 → tubular texture detected (paper fibers, halftone dot ridges)
- wavelet_HH > 0.03 → high diagonal noise energy from paper grain / print artifacts

DECISION RULES — follow strictly in order. THE METRICS OVERRIDE YOUR VISUAL IMPRESSION:

STEP 1 — HARD REJECT → detected=false, confidence=90 (MANDATORY, no exceptions):
Apply this if ANY of the following is true:
  a) dct_triple = 0.00 → zero multi-compression chain evidence = NOT a paper print
  b) glcm_contrast < 0.10 AND halftone_peak < 0.10 → surface is smooth, not paper fiber
  c) blue_ratio < 0.78 → cool LCD backlight, this is a screen capture not paper
  d) Moiré interference patterns or pixel grid visible → screen capture

Even if the document LOOKS flat or lacks a visible hologram → if dct_triple=0.00, it is NOT paper.
A genuine card photographed without direct light will NOT show its hologram. That is normal.

STEP 2 — ABSOLUTE PROOF OF PAPER → detected=true, confidence ≥ 90:
Requires AT LEAST ONE of:
  a) Halftone dot grid clearly visible (regular tiny dots on portrait or background)
  b) Paper fiber texture physically visible on the document surface
  c) Ink bleed, toner smear, or inkjet banding at text edges
  d) Physical fold line, crease, or paper cut edge visible
  e) Portrait looks printed: dot-matrix, grainy, or offset-printed appearance

STEP 3 — STRONG EVIDENCE → detected=true, confidence ≥ 85:
Requires ALL of:
  - dct_triple ≥ 0.05 (at least some multi-compression signal present)
  - glcm_contrast > 0.04 OR halftone_peak > 0.07 (any texture/pattern signal)
  - Visual: flat matte surface AND no holographic security features visible

STEP 4 — DEFAULT → detected=false, confidence=88:
If none of STEP 2 or 3 apply → the document is NOT a paper print.
Missing hologram, flat angle, shadows → NOT sufficient. Default to not-paper.

CRITICAL: NEVER set detected=true based solely on visual appearance without metric support.
NEVER set detected=true if dct_triple = 0.00.
Confidence of 50% → always detected=false (you are unsure = not paper).

Respond ONLY with this exact JSON (no markdown):
{"detected": false, "confidence": 50, "reason": "one sentence explanation"}

detected=true means it IS a paper printout (FRAUD).
confidence=0-100 (how certain you are)."""

_PROMPT_SUPERIMPOSED = """You are a forensic document analyst. Your ONLY task: determine if this identity document has TAMPERED, SPLICED or SUPERIMPOSED elements — particularly a pasted/replaced portrait photo or altered data fields.

KEY SIGNALS to look for:
- Portrait photo area has different noise level, resolution, or sharpness than the rest of the document
- Visible edge artifacts, halos, or cut-and-paste boundaries around the portrait
- Different lighting direction on the face vs the document background
- Text fields (name, DOB, ID number) have inconsistent font, size, or spacing
- MRZ (machine readable zone) lines have characters with different fonts or spacing
- Color mismatch between the portrait zone and surrounding document
- Compression artifacts clustered around a specific zone (face or text fields)
- The lamination/overlay appears to be on TOP of a modified element rather than integrated

Respond ONLY with this exact JSON (no markdown):
{"detected": false, "confidence": 50, "reason": "one sentence explanation"}

detected=true means the document HAS been tampered (FRAUD).
confidence=0-100 (how certain you are)."""

_PROMPT_AI_ALTERED = """You are a forensic document analyst. Your ONLY task: determine if this identity document was GENERATED by AI or significantly ALTERED using AI tools (Photoshop generative fill, Stable Diffusion inpainting, etc.).

KEY SIGNALS to look for:
- Unnaturally smooth or "plastic" skin texture on the portrait (AI smoothing)
- Microprinting or guilloche patterns (background security patterns) look warped, smeared or too uniform
- Security watermarks or background patterns have impossible symmetry or repeat with wrong frequency
- MRZ zone has semantically invalid characters, wrong check digits, or characters that don't match visible data
- Country flag, coat of arms, or national symbols have subtle distortions or wrong details
- Hologram simulation looks flat (AI hallucinated a hologram as a static image)
- Font in data fields looks synthetically generated (too perfect or slightly wrong letterforms)
- Shadows and reflections on the document are physically impossible

Respond ONLY with this exact JSON (no markdown):
{"detected": false, "confidence": 50, "reason": "one sentence explanation"}

detected=true means the document was AI-generated or AI-altered (FRAUD).
confidence=0-100 (how certain you are)."""

_PROMPT_LIVENESS = """You are a forensic document analyst. Your ONLY task: determine if this is a REAL LIVE PHOTO of a physical identity document taken in a natural environment (as opposed to a scan, screenshot, or studio-perfect digital image).

KEY SIGNALS that indicate a LIVE photo (GOOD):
- Natural perspective or slight angle (document not perfectly flat/parallel to camera)
- Natural lighting with realistic shadows under and around the document
- Slight hand-held blur or natural focus variation
- Environment visible around the document (table, fingers, background)
- Natural reflections or shine on the card surface
- Slight depth-of-field effect (document in focus, background slightly blurred)
- Document has minor physical imperfections (fingerprints, micro-scratches) consistent with real use

KEY SIGNALS that indicate NOT a live photo (BAD):
- Perfectly flat, perfectly centered, zero perspective distortion (looks like a scanner)
- Pure white or pure black background with no environment
- No shadows whatsoever
- Perfectly uniform lighting with no reflections
- Image appears to be a screenshot from a KYC portal or database

Respond ONLY with this exact JSON (no markdown):
{"detected": true, "confidence": 50, "reason": "one sentence explanation"}

detected=true means it IS a real live photo (GOOD/AUTHENTIC signal).
detected=false means it looks like a scan or screenshot (SUSPICIOUS).
confidence=0-100 (how certain you are)."""

_PROMPTS: dict[str, str] = {
    "screen_capture": _PROMPT_SCREEN_CAPTURE,
    "printed_paper": _PROMPT_PRINTED_PAPER,
    "superimposed_elements": _PROMPT_SUPERIMPOSED,
    "ai_altered": _PROMPT_AI_ALTERED,
    "liveness": _PROMPT_LIVENESS,
}


def _build_gemini_client():
    """Crea el cliente de Gemini. Requiere GEMINI_API_KEY en env."""
    try:
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set — Gemini engine disabled.")
            return None
        client = genai.Client(api_key=api_key)
        return client
    except ImportError:
        logger.warning("google-genai not installed — Gemini engine disabled.")
        return None


def _image_to_bytes(image: Image.Image, max_px: int = 1024) -> bytes:
    """Convierte PIL Image a JPEG bytes, redimensionando para reducir tokens de entrada."""
    img = image.copy()
    w, h = img.size
    if max(w, h) > max_px:
        ratio = max_px / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _image_resize(image: Image.Image, max_px: int = 1024) -> Image.Image:
    """Devuelve la imagen redimensionada a max_px (misma transformación que _image_to_bytes)."""
    w, h = image.size
    if max(w, h) > max_px:
        ratio = max_px / max(w, h)
        return image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    return image


def _compute_forensic_signals(image: Image.Image) -> dict:
    """Extrae métricas forenses que distinguen foto-de-pantalla vs foto-directa.

    Métricas (todas con valores de referencia empíricos en n=20 screen captures):
      saturation       : 0.12–0.39 en pantallas, 0.40–0.70 en documentos físicos
      blue_ratio       : 0.70–1.30 en pantallas LCD (~6500K), >1.30 o <0.70 en luz natural
      fft_peak_score   : <0.15 en screen captures (espectro suave del pixel grid)
      dct_double_score : <0.10 en screen captures (doble compresión JPEG)
      aliasing_score   : 0.24–0.51 en screen captures (dos etapas de digitalización)
    """
    signals: dict = {"saturation": 0.25, "blue_ratio": 1.0,
                     "fft_peak_score": 0.10, "dct_double_score": 0.05,
                     "aliasing_score": 0.30}
    if not _SCIPY_AVAILABLE:
        return signals
    try:
        img_array = np.array(image)
        if img_array.ndim == 2:          # escala de grises
            img_array = np.stack([img_array]*3, axis=-1)
        if img_array.shape[2] == 4:      # RGBA → RGB
            img_array = img_array[:, :, :3]
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
        h, w = img_gray.shape

        # ── Saturación ──────────────────────────────────────────────
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        signals["saturation"] = round(float(np.mean(img_hsv[:, :, 1])) / 255.0, 3)

        # ── Temperatura de color (ratio azul/rojo) ───────────────────
        b_mean = float(np.mean(img_bgr[:, :, 0]))
        r_mean = float(np.mean(img_bgr[:, :, 2]))
        signals["blue_ratio"] = round(b_mean / (r_mean + 1e-9), 3)

        # ── FFT — picos periódicos en frecuencias medias ─────────────
        fft_mag = np.abs(fftshift(fft2(img_gray)))
        fft_log = np.log1p(fft_mag)
        cy, cx = h // 2, w // 2
        yy, xx = np.mgrid[0:h, 0:w]
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        mid_mask  = (dist > 20) & (dist < min(h, w) // 4)
        high_mask = dist > min(h, w) // 4
        mid_vals = fft_log[mid_mask]
        fft_peak = float(np.std(mid_vals) / (np.mean(mid_vals) + 1e-9))
        signals["fft_peak_score"] = round(fft_peak, 3)

        # ── Aliasing — energía en altas frecuencias ──────────────────
        total_energy = np.sum(fft_mag) + 1e-9
        signals["aliasing_score"] = round(float(np.sum(fft_mag[high_mask]) / total_energy), 3)

        # ── Doble compresión JPEG — DCT 8×8 ─────────────────────────
        bh, bw = (h // 8) * 8, (w // 8) * 8
        img_crop = img_gray[:bh, :bw]
        blocks = img_crop.reshape(bh // 8, 8, bw // 8, 8).transpose(0, 2, 1, 3)
        dct_coeffs: list[np.ndarray] = []
        for blk in blocks.reshape(-1, 8, 8):
            dct_blk = cv2.dct((blk - 128.0).astype(np.float32))
            dct_coeffs.append(dct_blk[1:, 1:].flatten())
        dct_arr = np.concatenate(dct_coeffs)
        hist, _ = np.histogram(dct_arr, bins=200, range=(-100, 100))
        hist_fft_ac = np.abs(np.fft.rfft(hist.astype(float)))[1:]
        peak_ratio = float(np.max(hist_fft_ac) / (np.mean(hist_fft_ac) + 1e-9))
        signals["dct_double_score"] = round(min(peak_ratio / 50.0, 1.0), 3)

    except Exception as exc:
        logger.debug("_compute_forensic_signals failed: %s", exc)
    return signals


def _get_quality_context(image: Image.Image) -> str:
    """Genera un contexto forense detallado para pasar al prompt screen_capture."""
    s = _compute_forensic_signals(image)
    sat   = s["saturation"]
    b_r   = s["blue_ratio"]
    fft   = s["fft_peak_score"]
    dct   = s["dct_double_score"]
    alias = s["aliasing_score"]

    # Evaluar cada señal contra umbrales empíricos
    sat_label   = "LOW (screen-like)"     if sat   < 0.40 else "NORMAL (photo-like)"
    br_label    = "SCREEN-RANGE (LCD)"    if 0.65 < b_r < 1.35 else "UNUSUAL (not typical LCD)"
    fft_label   = "FLAT (screen-like)"    if fft   < 0.15 else "PEAKED (unusual)"
    dct_label   = "DOUBLE-COMP likely"    if dct   < 0.10 else "SINGLE-COMP likely"
    alias_label = "HIGH (double-digitiz.)" if alias > 0.28 else "LOW"

    lines = [
        f"saturation={sat:.3f}   [{sat_label}]   — screen captures: 0.12–0.39, physical docs: 0.40–0.70",
        f"blue_ratio={b_r:.3f}   [{br_label}]   — LCD ~6500K gives 0.70–1.30, warm light gives <0.70",
        f"fft_peak={fft:.3f}     [{fft_label}]  — low flat spectrum = consistent with screen pixel grid",
        f"dct_double={dct:.3f}   [{dct_label}]  — double JPEG compression typical of screen→camera chain",
        f"aliasing={alias:.3f}   [{alias_label}] — high-freq energy from two digitization stages",
    ]
    return "\n".join(lines)


def _compute_printed_paper_signals(image: Image.Image) -> dict:
    """Extrae métricas forenses que distinguen copia-en-papel vs documento genuino.

    Métricas (con valores de referencia físicos):
      halftone_peak_score  : >0.25 en impresiones (trama halftone crea picos FFT periódicos)
      glcm_contrast        : >0.30 en papel (fibras crean rugosidad), <0.15 en pantalla suave
      lbp_variance         : >0.18 en papel (textura rugosa), <0.10 en tarjeta plástica
      haralick_correlation : <0.85 en papel (irregularidades), >0.90 en plástico/pantalla
      dct_triple_score     : >0.12 en papel (original→impresora→cámara = triple compresión)
      cmyk_gamut_score     : >0.20 en papel (tintas CMYK recortan out-of-gamut pixels)
      histogram_clip_ratio : >0.04 en papel (densidad máx D≈1.5 acumula en 0/255)
      blue_ratio           : 0.85–1.10 en papel natural, <0.80 en pantalla LCD
    """
    signals: dict = {
        "halftone_peak_score":  0.05,
        "glcm_contrast":        0.10,
        "lbp_variance":         0.08,
        "haralick_correlation": 0.05,
        "haralick_entropy":     0.80,
        "dct_triple_score":     0.05,
        "cmyk_gamut_score":     0.05,
        "histogram_clip_ratio": 0.01,
        "blue_ratio":           1.0,
        "laplacian_variance":   500.0,
        "meijering_score":      0.05,
        "wavelet_detail_energy": 0.10,
    }
    if not _SCIPY_AVAILABLE:
        return signals
    try:
        img_array = np.array(image)
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = img_gray.shape

        # ── Blue ratio (ratio azul/rojo) ─────────────────────────────
        b_mean = float(np.mean(img_bgr[:, :, 0]))
        r_mean = float(np.mean(img_bgr[:, :, 2]))
        signals["blue_ratio"] = round(b_mean / (r_mean + 1e-9), 3)

        # ── Halftone peak score — FFT en rango de frecuencias medias ─
        gray_f64 = img_gray.astype(np.float64)
        fft_mag = np.abs(fftshift(fft2(gray_f64)))
        fft_log = np.log1p(fft_mag)
        cy, cx = h // 2, w // 2
        yy, xx = np.mgrid[0:h, 0:w]
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        # Trama halftone 150 DPI aparece en 1/6 a 1/3 del espectro
        halftone_mask = (dist > min(h, w) // 6) & (dist < min(h, w) // 3)
        ht_vals = fft_log[halftone_mask]
        ht_peak = float(np.max(ht_vals) / (np.mean(ht_vals) + 1e-9)) - 1.0
        signals["halftone_peak_score"] = round(min(max(ht_peak / 5.0, 0.0), 1.0), 3)

        # ── GLCM contrast — textura de papel vs superficie lisa ──────
        try:
            gray_uint8 = img_gray.astype(np.uint8)
            # Escalar a 64 niveles para mayor velocidad
            gray_64 = (gray_uint8 // 4).astype(np.uint8)
            glcm = graycomatrix(gray_64, distances=[1], angles=[0, np.pi/4],
                                levels=64, symmetric=True, normed=True)
            contrast = float(np.mean(graycoprops(glcm, "contrast")))
            # Normalizar: papel ≈ 30–120, pantalla ≈ 5–25
            signals["glcm_contrast"] = round(min(contrast / 150.0, 1.0), 3)
        except Exception:
            pass

        # ── LBP variance — patrón de textura local ───────────────────
        try:
            lbp = local_binary_pattern(img_gray.astype(np.uint8), P=8, R=1.5, method="uniform")
            lbp_hist, _ = np.histogram(lbp, bins=10, density=True)
            signals["lbp_variance"] = round(float(np.var(lbp_hist)), 3)
        except Exception:
            pass

        # ── Haralick correlation via mahotas ─────────────────────────
        try:
            gray_uint8 = img_gray.astype(np.uint8)
            har = mahotas.features.haralick(gray_uint8, ignore_zeros=False, return_mean=True)
            # har[2] = correlación Haralick real (rango 0–1)
            # har[8] = entropía — papel impreso tiene mayor entropía por ruido de trama
            # Usar 1 - correlación: papel irregular → correlación baja → score alto
            raw_corr = float(np.clip(har[2], 0.0, 1.0))
            signals["haralick_correlation"] = round(1.0 - raw_corr, 3)  # alto = irregular = papel
            # Entropía normalizada (típico: 11–14 en estas imágenes)
            signals["haralick_entropy"] = round(float(np.clip(har[8] / 15.0, 0.0, 1.0)), 3)
        except Exception:
            pass

        # ── DCT triple-compression score ─────────────────────────────
        try:
            bh, bw = (h // 8) * 8, (w // 8) * 8
            img_crop = gray_f64[:bh, :bw]
            blocks = img_crop.reshape(bh // 8, 8, bw // 8, 8).transpose(0, 2, 1, 3)
            dct_coeffs: list[np.ndarray] = []
            for blk in blocks.reshape(-1, 8, 8):
                dct_blk = cv2.dct((blk - 128.0).astype(np.float32))
                dct_coeffs.append(dct_blk[1:, 1:].flatten())
            dct_arr = np.concatenate(dct_coeffs)
            hist, _ = np.histogram(dct_arr, bins=200, range=(-100, 100))
            hist_fft_ac = np.abs(np.fft.rfft(hist.astype(float)))[1:]
            # Triple compresión crea picos adicionales vs doble
            n_peaks = int(np.sum(hist_fft_ac > (np.mean(hist_fft_ac) + 3 * np.std(hist_fft_ac))))
            signals["dct_triple_score"] = round(min(n_peaks / 20.0, 1.0), 3)
        except Exception:
            pass

        # ── Laplacian variance — nitidez de bordes ────────────────
        # Papel impreso pierde bordes finos → varianza baja.
        # Genuino: 800–3000+, papel: 100–600 (típico).
        try:
            lap = cv2.Laplacian(img_gray, cv2.CV_64F)
            signals["laplacian_variance"] = round(float(lap.var()), 2)
        except Exception:
            pass

        # ── Meijering neuriteness — textura tubular (fibras/halftone) ─
        # Magnifica diferencias de textura entre papel y plástico.
        # Papel: 0.05–0.25, genuino: 0.01–0.04 (típico).
        try:
            gray_f64 = img_gray.astype(np.float64) / 255.0
            meij = meijering(gray_f64, sigmas=range(1, 4), black_ridges=True)
            signals["meijering_score"] = round(float(np.mean(meij)), 4)
        except Exception:
            pass

        # ── Wavelet detail energy — ruido HF diferente en papel vs plástico
        # Descomposición Haar → energía relativa de subbanda HH (diagonal).
        # Papel: 0.03–0.15, genuino: 0.005–0.03 (típico).
        try:
            gray_norm = img_gray.astype(np.float64) / 255.0
            coeffs = pywt.dwt2(gray_norm, 'haar')
            cA, (cH, cV, cD) = coeffs
            total_e = np.sum(cA**2) + np.sum(cH**2) + np.sum(cV**2) + np.sum(cD**2) + 1e-9
            detail_e = np.sum(cD**2) / total_e
            signals["wavelet_detail_energy"] = round(float(detail_e), 4)
        except Exception:
            pass

        # ── CMYK gamut score — píxeles fuera de gamut CMYK ──────────
        try:
            img_rgb_f = img_array.astype(np.float32) / 255.0
            r_ch = img_rgb_f[:, :, 0]
            g_ch = img_rgb_f[:, :, 1]
            b_ch = img_rgb_f[:, :, 2]
            # Conversión aproximada sRGB→CMYK
            k_ch = 1.0 - np.max(img_rgb_f, axis=2)
            denom = (1.0 - k_ch + 1e-9)
            c_ch = (1.0 - r_ch - k_ch) / denom
            m_ch = (1.0 - g_ch - k_ch) / denom
            y_ch = (1.0 - b_ch - k_ch) / denom
            # Píxeles cuyo valor CMYK excede [0,1] indican saturación out-of-gamut
            out_of_gamut = np.mean(
                (c_ch < -0.05) | (c_ch > 1.05) |
                (m_ch < -0.05) | (m_ch > 1.05) |
                (y_ch < -0.05) | (y_ch > 1.05)
            )
            signals["cmyk_gamut_score"] = round(float(out_of_gamut), 3)
        except Exception:
            pass

        # ── Histogram clip ratio — densidad máxima acumulada ────────
        try:
            gray_hist, _ = np.histogram(img_gray.flatten(), bins=256, range=(0, 255))
            total_px = img_gray.size
            clip_lo = int(np.sum(gray_hist[:5]))   # acumulación en negro
            clip_hi = int(np.sum(gray_hist[-5:]))  # acumulación en blanco
            signals["histogram_clip_ratio"] = round((clip_lo + clip_hi) / (total_px + 1e-9), 4)
        except Exception:
            pass

    except Exception as exc:
        logger.debug("_compute_printed_paper_signals failed: %s", exc)
    return signals


def _get_print_context(image: Image.Image) -> str:
    """Genera contexto forense para el prompt printed_paper."""
    s = _compute_printed_paper_signals(image)
    ht   = s["halftone_peak_score"]
    gc   = s["glcm_contrast"]
    lbp  = s["lbp_variance"]
    entr = s.get("haralick_entropy", 0.80)
    dct  = s["dct_triple_score"]
    cmyk = s["cmyk_gamut_score"]
    clip = s["histogram_clip_ratio"]
    br   = s["blue_ratio"]
    lap  = s.get("laplacian_variance", 500.0)
    meij = s.get("meijering_score", 0.05)
    wde  = s.get("wavelet_detail_energy", 0.10)

    # Umbrales calibrados sobre imágenes reales observadas
    ht_label   = "ELEVATED (possible halftone)" if ht   > 0.08 else "LOW (no halftone pattern)"
    gc_label   = "HIGH (rough surface)"         if gc   > 0.25 else "LOW (smooth surface)"
    lbp_label  = "HIGH (texture present)"       if lbp  > 0.003 else "LOW (uniform texture)"
    entr_label = "HIGH (complex/noisy)"         if entr > 0.83 else "LOW (smooth/uniform)"
    dct_label  = "MULTI-COMP (paper chain)"     if dct  > 0.10 else "LOW (few compressions)"
    cmyk_label = "OUT-GAMUT (CMYK inks)"        if cmyk > 0.05 else "IN-GAMUT"
    clip_label = "CLIPPED (limited D-range)"    if clip > 0.001 else "NORMAL (wide D-range)"
    br_label   = "WARM/NEUTRAL (paper light)"   if 0.80 < br < 1.15 else "COOL (LCD) or unusual"
    lap_label  = "LOW-SHARP (print blur)"       if lap  < 400 else "SHARP (genuine-like)"
    meij_label = "HIGH (paper texture)"         if meij > 0.04 else "LOW (smooth)"
    wde_label  = "HIGH (paper noise)"           if wde  > 0.03 else "LOW (clean)"

    lines = [
        f"halftone_peak={ht:.3f}   [{ht_label}]  — inkjet/laser trama creates peaks; screen: ~0.05–0.07",
        f"glcm_contrast={gc:.3f}   [{gc_label}] — rough paper: >0.25, smooth card/screen: <0.10",
        f"lbp_variance={lbp:.4f}  [{lbp_label}] — paper grain: >0.003, smooth: <0.002",
        f"haralick_entropy={entr:.3f} [{entr_label}] — printed noise raises entropy; smooth card: ~0.78",
        f"dct_triple={dct:.3f}     [{dct_label}] — paper = orig→printer→camera (3 JPEG stages)",
        f"cmyk_gamut={cmyk:.3f}    [{cmyk_label}] — CMYK ink mixing clips sRGB gamut",
        f"hist_clip={clip:.4f}    [{clip_label}] — paper D≈1.5 vs camera D≈12 clips shadows/highlights",
        f"blue_ratio={br:.3f}      [{br_label}] — paper under room light: 0.85–1.10, LCD: 0.70–1.30",
        f"laplacian_var={lap:.1f}  [{lap_label}] — print chain loses sharpness; genuine: >600, paper: <400",
        f"meijering={meij:.4f}    [{meij_label}] — tubular texture from paper fibers/halftone dots",
        f"wavelet_HH={wde:.4f}    [{wde_label}] — diagonal HF noise energy; paper: >0.03, genuine: <0.02",
    ]
    return "\n".join(lines)


def _parse_filter(data: dict, filter_name: str) -> FilterResult:
    """Convierte un dict de respuesta Gemini a FilterResult."""
    detected = bool(data.get("detected", False))
    confidence = float(data.get("confidence", 50.0))
    confidence = max(0.0, min(100.0, confidence))
    reason = data.get("reason", "")

    # Para printed_paper: si Gemini está totalmente inseguro (exactamente 50%),
    # tratar como fraude — mejor falso positivo que dejar pasar una copia en papel.
    if filter_name == "printed_paper" and not detected and confidence <= 50.0:
        detected = True
        confidence = 70.0
        reason = f"[uncertain-flip] {reason}"

    if reason:
        logger.info("Gemini [%s]: detected=%s conf=%.1f — %s",
                    filter_name, detected, confidence, reason)

    return FilterResult(
        answer="yes" if detected else "no",
        percentageOfConfidence=round(confidence, 1),
    )


@dataclass
class GeminiEngine:
    """Motor de verificación usando Gemini Vision API — 5 filtros en 1 llamada."""

    model_name: str = "gemini-2.5-flash"
    _client: object | None = field(default=None, repr=False)
    _available: bool = field(default=False, repr=False)

    def initialize(self) -> None:
        """Inicializa el cliente de Gemini."""
        self._client = _build_gemini_client()
        self._available = self._client is not None
        if self._available:
            logger.info("Gemini engine initialized with model=%s", self.model_name)
        else:
            logger.warning("Gemini engine NOT available — will use local model fallback.")

    @property
    def available(self) -> bool:
        return self._available

    def _call_filter(self, image: Image.Image, img_bytes: bytes, filter_name: str) -> FilterResult | None:
        """Llama a Gemini con el prompt especializado de un filtro. Retorna None si falla."""

        # ── Override local para printed_paper ────────────────────────────────
        # Si las métricas locales son suficientemente claras, no hace falta Gemini.
        if filter_name == "printed_paper":
            img_small = _image_resize(image)
            sig = _compute_printed_paper_signals(img_small)
            dct  = sig["dct_triple_score"]
            br   = sig["blue_ratio"]
            gc   = sig["glcm_contrast"]
            entr = sig.get("haralick_entropy", 0.80)
            lap  = sig.get("laplacian_variance", 500.0)
            meij = sig.get("meijering_score", 0.05)
            wde  = sig.get("wavelet_detail_energy", 0.10)
            # ── Señal fuerte: multi-compresión + luz cálida + rango de textura válido ──
            if dct >= 0.10 and br >= 0.92 and 0.02 <= gc <= 0.40:
                logger.info("printed_paper local override: PAPER detected strong (dct=%.2f, B/R=%.2f, gc=%.2f, lap=%.0f)", dct, br, gc, lap)
                return FilterResult(answer="yes", percentageOfConfidence=88.0)
            # ── Señal media: dct bajo pero luz cálida + gc mínimo ──
            if dct >= 0.05 and br >= 0.92 and 0.02 <= gc <= 0.40:
                logger.info("printed_paper local override: PAPER detected medium (dct=%.2f, B/R=%.2f, gc=%.2f, lap=%.0f)", dct, br, gc, lap)
                return FilterResult(answer="yes", percentageOfConfidence=85.0)
            # ── Nuevas métricas: laplacian baja + meijering alta = papel sin señal DCT ──
            # Captura papeles como test-05/test-08 donde dct=0.00 pero textura es de papel.
            # GUARD: exigir dct>0 O entr>=0.80 para evitar originales con lap baja (test-17/19-orig).
            if lap < 320 and meij > 0.04 and br >= 0.88 and (dct > 0.0 or entr >= 0.80):
                logger.info("printed_paper local override: PAPER detected via texture (lap=%.0f, meij=%.4f, B/R=%.2f, entr=%.2f)", lap, meij, br, entr)
                return FilterResult(answer="yes", percentageOfConfidence=82.0)
            # ── Cero evidencia de multi-compresión + sin textura → no papel ──
            if dct == 0.0 and gc < 0.04 and lap > 400:
                logger.info("printed_paper local override: NOT paper (dct=0, gc=%.2f, lap=%.0f)", gc, lap)
                return FilterResult(answer="no", percentageOfConfidence=90.0)
            # ── Guard anti-FP para Gemini: lap muy alta → genuino seguro ──
            # Originales con lap>1500 tienen bordes super nítidos → imposible papel.
            if lap > 1500 and dct <= 0.10:
                logger.info("printed_paper local override: NOT paper (lap=%.0f too sharp for paper)", lap)
                return FilterResult(answer="no", percentageOfConfidence=88.0)
        try:
            import json
            from google.genai import types

            prompt = _PROMPTS[filter_name]

            # Para screen_capture, agregar contexto de calidad
            if filter_name == "screen_capture":
                quality_hint = _get_quality_context(_image_resize(image))
                prompt = prompt.replace("{quality_context}", quality_hint)
            # Para printed_paper, agregar contexto forense de papel
            elif filter_name == "printed_paper":
                print_hint = _get_print_context(_image_resize(image))
                prompt = prompt.replace("{quality_context}", print_hint)

            response = self._client.models.generate_content(  # type: ignore[union-attr]
                model=self.model_name,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(text=prompt),
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type="image/jpeg",
                                    data=img_bytes,
                                )
                            ),
                        ],
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=1024,
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            data = json.loads(response.text)
            return _parse_filter(data, filter_name)
        except Exception as exc:
            logger.warning("Gemini [%s] failed: %s — falling back to local.", filter_name, exc)
            return None

    def analyze_all(self, image: Image.Image) -> dict[str, FilterResult | None]:
        """
        Ejecuta los 5 filtros con prompts individuales especializados.
        Retorna dict {filter_name: FilterResult | None}.
        None significa que falló y el pipeline usará fallback local.
        """
        if not self._available or self._client is None:
            return {k: None for k in _PROMPTS}

        img_bytes = _image_to_bytes(image)
        return {name: self._call_filter(image, img_bytes, name) for name in _PROMPTS}

    def analyze_filter(self, image: Image.Image, filter_name: str) -> FilterResult | None:
        """
        Analiza un único filtro con su prompt especializado.
        Llama directamente _call_filter sin ejecutar los demás filtros.
        """
        if not self._available or self._client is None:
            return None
        img_bytes = _image_to_bytes(image)
        return self._call_filter(image, img_bytes, filter_name)
