"""
Pre-análisis forense local (numpy + PIL) para enriquecer el prompt de Gemini.

Inspirado en CheckScan (FFT magnitude peaks), FHAG (halftone band energy)
y MMDT (multi-modal disentangled traces) — pero sin tomar decisiones por sí
mismo: devuelve sólo medidas objetivas que Gemini incorpora en su razonamiento.

Las 4 features son baratas (<50 ms en CPU para una imagen 1024×768):
  • halftone_score       — energía FFT en banda anular típica de print CMYK 150 lpi
  • specular_count       — número de "destellos" pequeños (laminado polycarbonato)
  • edge_sharpness       — varianza del Laplaciano en el borde del documento
  • color_saturation_p95 — percentil 95 de la saturación HSV (laser engraving = vivo)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


# Banda anular en frecuencia donde aparece el ruido de halftone CMYK
# (calibrado para imágenes ~512 px de lado, ajustado dinámicamente abajo).
_HALFTONE_BAND_LOW_FRAC = 0.18   # fracción del radio máximo
_HALFTONE_BAND_HIGH_FRAC = 0.45


@dataclass(frozen=True)
class ForensicFeatures:
    halftone_score: float          # 0..1 — alto = patrón de impresión
    specular_count: int            # número de destellos (>240) detectados
    edge_sharpness: float          # varianza del Laplaciano en bordes
    color_saturation_p95: float    # 0..1
    # Veredictos cualitativos derivados (para el prompt)
    halftone_verdict: str          # "low" | "medium" | "high"
    specular_verdict: str          # "none" | "few" | "many"
    saturation_verdict: str        # "washed" | "normal" | "vivid"

    def as_prompt_block(self) -> str:
        # Heurística rápida: combinación clásica de paper printout
        printout_alert = (
            self.color_saturation_p95 < 0.30
            and self.specular_count <= 1
        )
        alert_line = (
            "  ⚠ PRINTOUT ALERT: washed colors + no specular highlights — this combination "
            "is the signature of a paper photocopy. Do NOT override with subjective "
            "impressions of 'thickness' or 'shadows'.\n"
            if printout_alert
            else ""
        )
        return (
            "════════════════════════════════════════════════════════════════════\n"
            "FORENSIC PRE-ANALYSIS — objective pixel-level measurements\n"
            "(computed by a deterministic numpy filter, NOT subjective):\n"
            "════════════════════════════════════════════════════════════════════\n"
            f"  halftone_score:       {self.halftone_score:.2f}  [{self.halftone_verdict.upper()}]   "
            "high → CMYK halftone pattern typical of paper print\n"
            f"  specular_highlights:  {self.specular_count:>3d}      [{self.specular_verdict.upper()}]   "
            "real polycarbonate lamination shows several small bright spots; paper shows ~0\n"
            f"  edge_sharpness:       {self.edge_sharpness:7.1f}            "
            "low values (<300) suggest a flat paper edge\n"
            f"  color_saturation_p95: {self.color_saturation_p95:.2f}  [{self.saturation_verdict.upper()}]   "
            "WASHED (<0.30) → photocopy/printout; VIVID (>0.55) → real laser-engraved card\n"
            f"{alert_line}"
            "════════════════════════════════════════════════════════════════════\n"
            "RULE: trust these numbers over your visual impression of 'thickness',\n"
            "'shadows' or 'natural wear', because those signals are easily faked by\n"
            "a photo of a printout on a table. The numbers above measure properties\n"
            "of the IMAGE itself that cannot be faked by re-photographing.\n"
            "════════════════════════════════════════════════════════════════════\n"
        )


def _to_gray_512(image: Image.Image) -> np.ndarray:
    """Resize a 512 px lado mayor manteniendo aspect, devolver uint8 grayscale."""
    img = image.convert("L")
    w, h = img.size
    scale = 512.0 / max(w, h)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    img = img.resize(new_size, Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def _halftone_score(gray: np.ndarray) -> float:
    """
    Detecta picos estrechos en el perfil radial del espectro — firma típica
    del halftone CMYK (frecuencia espacial regular). Una foto natural muestra
    un decaimiento 1/f suave sin picos.

    Devuelve un valor ~0..1: cuánto sobresale el pico máximo en la banda
    de halftone respecto al fondo 1/f local.
    """
    g = gray - float(gray.mean())
    # Ventana Hann para evitar leakage espectral en los bordes
    h, w = g.shape
    wy = np.hanning(h)[:, None]
    wx = np.hanning(w)[None, :]
    g = g * (wy * wx)

    fft = np.fft.fft2(g)
    mag = np.abs(np.fft.fftshift(fft))
    # log para comprimir el rango dinámico (DC domina si no)
    mag = np.log1p(mag)

    cy, cx = h / 2.0, w / 2.0
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_max = float(min(cy, cx))

    # Perfil radial promedio — bins de 1 px
    r_int = r.astype(np.int32)
    r_int_flat = r_int.ravel()
    mag_flat = mag.ravel()
    n_bins = int(r_max)
    if n_bins < 20:
        return 0.0
    radial = np.bincount(r_int_flat, weights=mag_flat, minlength=n_bins + 1)[: n_bins + 1]
    counts = np.bincount(r_int_flat, minlength=n_bins + 1)[: n_bins + 1]
    radial = radial / np.maximum(counts, 1)

    lo = int(_HALFTONE_BAND_LOW_FRAC * r_max)
    hi = int(_HALFTONE_BAND_HIGH_FRAC * r_max)
    if hi - lo < 5:
        return 0.0

    band = radial[lo:hi]
    # Tendencia local (smooth) restada — buscamos PICOS, no energía absoluta
    win = max(5, (hi - lo) // 6)
    kernel = np.ones(win) / win
    trend = np.convolve(band, kernel, mode="same")
    residual = band - trend

    peak = float(residual.max())
    noise = float(np.std(residual) + 1e-8)
    snr = peak / noise  # SNR del pico más alto

    # SNR típico: foto natural ~1.5-2.5, halftone fuerte 4-8.
    return float(np.clip((snr - 2.0) / 5.0, 0.0, 1.0))


def _specular_count(image: Image.Image) -> int:
    """
    Cuenta blobs muy brillantes (>=245) y pequeños (<0.5% del área),
    típicos del reflejo del laminado polycarbonato.
    """
    arr = np.asarray(image.convert("L").resize((512, 512)), dtype=np.uint8)
    bright = arr >= 245
    if not bright.any():
        return 0
    # Connected-components rudimentario por flood-fill iterativo barato:
    # contamos transiciones en filas + columnas como proxy de "regiones".
    # Para no añadir scipy, usamos una aproximación: número de "runs" 4-vecinos.
    visited = np.zeros_like(bright, dtype=bool)
    count = 0
    h, w = bright.shape
    max_blob = int(0.005 * h * w)  # 0.5% del área = blob "pequeño"

    # BFS iterativo con stack — barato porque bright suele ser muy esparso
    for i in range(h):
        for j in range(w):
            if not bright[i, j] or visited[i, j]:
                continue
            # Flood-fill
            stack = [(i, j)]
            size = 0
            while stack and size <= max_blob + 1:
                ci, cj = stack.pop()
                if ci < 0 or ci >= h or cj < 0 or cj >= w:
                    continue
                if visited[ci, cj] or not bright[ci, cj]:
                    continue
                visited[ci, cj] = True
                size += 1
                stack.append((ci + 1, cj))
                stack.append((ci - 1, cj))
                stack.append((ci, cj + 1))
                stack.append((ci, cj - 1))
            if 3 <= size <= max_blob:
                count += 1
            if count > 50:  # cap para no contar fondos brillantes enteros
                return count
    return count


def _edge_sharpness(gray: np.ndarray) -> float:
    """Varianza del Laplaciano sólo en píxeles con gradiente fuerte."""
    # Laplaciano discreto 3x3 sin scipy
    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    # Gradiente para máscara
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx[1:-1, 1:-1] ** 2 + gy[1:-1, 1:-1] ** 2)
    mask = grad > np.percentile(grad, 90)
    if not mask.any():
        return float(lap.var())
    return float(lap[mask].var())


def _color_saturation_p95(image: Image.Image) -> float:
    """Percentil 95 de la saturación HSV ∈ [0,1]."""
    arr = np.asarray(image.convert("HSV").resize((256, 256)), dtype=np.float32)
    sat = arr[..., 1] / 255.0
    return float(np.percentile(sat, 95))


def compute_features(image: Image.Image) -> ForensicFeatures:
    """Calcula las 4 features y las verbaliza para el prompt."""
    gray = _to_gray_512(image)

    halftone = _halftone_score(gray)
    specular = _specular_count(image)
    edges = _edge_sharpness(gray)
    sat = _color_saturation_p95(image)

    halftone_v = "high" if halftone >= 0.55 else "medium" if halftone >= 0.30 else "low"
    specular_v = "many" if specular >= 4 else "few" if specular >= 1 else "none"
    sat_v = "vivid" if sat >= 0.55 else "normal" if sat >= 0.30 else "washed"

    return ForensicFeatures(
        halftone_score=halftone,
        specular_count=specular,
        edge_sharpness=edges,
        color_saturation_p95=sat,
        halftone_verdict=halftone_v,
        specular_verdict=specular_v,
        saturation_verdict=sat_v,
    )
