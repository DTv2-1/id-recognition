"""
Script para descargar los pesos de los modelos necesarios.
Ejecutar una vez antes de construir el Docker image.

Uso:
    python scripts/download_weights.py
"""

import logging
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Pesos disponibles para descarga ────────────────────────────
DOWNLOADS = {
    # UnivFD fc weights — del repositorio oficial
    # https://github.com/WisconsinAIVision/UniversalFakeDetect
    # Copiar pretrained_weights/fc_weights.pth → weights/univfd_fc_weights.pth
    "univfd_fc_weights.pth": None,  # Manual: copiar del repo clonado

    # MiniFASNetV2 — exportar de Silent-Face-Anti-Spoofing a ONNX
    # https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
    "minifasnetv2_antispoof.onnx": None,  # Manual: exportar con script

    # Screen Capture CMA — entrenar con SIDTD + DLC-2021
    "screen_capture_cma.pth": None,  # Entrenar

    # Print Detection — entrenar con DLC-2021
    "print_detection_effnet.pth": None,  # Entrenar

    # Forgery Detection — entrenar con SIDTD/DocTamper
    "forgery_hifi.pth": None,  # Entrenar o descargar de HiFi-IFDL repo
}


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        logger.info("Already exists: %s", dest)
        return
    logger.info("Downloading %s → %s", url, dest)
    urllib.request.urlretrieve(url, dest)
    logger.info("Done: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)


def main() -> None:
    logger.info("Weights directory: %s", WEIGHTS_DIR)

    for filename, url in DOWNLOADS.items():
        dest = WEIGHTS_DIR / filename
        if url is not None:
            download_file(url, dest)
        else:
            if dest.exists():
                logger.info("✓ %s found", filename)
            else:
                logger.warning(
                    "✗ %s not found — needs manual setup. See comments in script.",
                    filename,
                )

    logger.info("=== Weight status ===")
    for filename in DOWNLOADS:
        status = "✓" if (WEIGHTS_DIR / filename).exists() else "✗ MISSING"
        logger.info("  %s %s", status, filename)


if __name__ == "__main__":
    main()
