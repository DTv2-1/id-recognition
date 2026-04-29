#!/usr/bin/env python3
"""
ADAMO ID — CLI de verificación de cédulas.

Permite probar el pipeline completo con cualquier imagen o carpeta sin
levantar la API. Usa el motor real (Gemini Vision + análisis forense
local) y devuelve los 5 filtros + el veredicto consolidado.

USO:
    # Una imagen
    python scripts/verify.py path/to/cedula.jpg

    # Toda una carpeta (paraleliza con 4 workers por defecto)
    python scripts/verify.py path/to/carpeta/

    # Carpeta recursiva (incluir subcarpetas)
    python scripts/verify.py path/to/carpeta/ --recursive

    # Cambiar paralelismo
    python scripts/verify.py path/to/carpeta/ --workers 8

    # Salida JSON (para integrar con otros scripts)
    python scripts/verify.py path/to/cedula.jpg --json

    # Guardar reporte JSON a archivo
    python scripts/verify.py path/to/carpeta/ --output reporte.json

REQUISITOS:
    - Variable GEMINI_API_KEY en el archivo .env
    - Formatos soportados: JPG, JPEG, PNG, WEBP
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Asegurar que el raíz del proyecto está en el path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from PIL import Image  # noqa: E402

from app.filters.gemini_engine import GeminiEngine, FILTER_NAMES  # noqa: E402
from app.schemas import FilterResult  # noqa: E402

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Códigos ANSI para colorear la salida (solo si stdout es TTY)
USE_COLOR = sys.stdout.isatty()
def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text
GREEN  = lambda s: _c("32", s)
RED    = lambda s: _c("31", s)
YELLOW = lambda s: _c("33", s)
CYAN   = lambda s: _c("36", s)
BOLD   = lambda s: _c("1", s)
DIM    = lambda s: _c("2", s)


# ─────────────────────────────────────────────────────────────────────
# Verificación de una imagen
# ─────────────────────────────────────────────────────────────────────
def verify_image(image_path: Path, engine: GeminiEngine) -> dict:
    """Procesa una imagen y devuelve el resultado completo."""
    t0 = time.time()
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        return {
            "file": str(image_path),
            "status": "error",
            "error": f"No se pudo abrir la imagen: {exc}",
            "elapsed_ms": int((time.time() - t0) * 1000),
        }

    try:
        results = engine.analyze_all(image)
    except Exception as exc:
        return {
            "file": str(image_path),
            "status": "error",
            "error": f"Falla del motor: {exc}",
            "elapsed_ms": int((time.time() - t0) * 1000),
        }

    # Rellenar None con neutro
    def _or_neutral(r: FilterResult | None) -> FilterResult:
        return r if r is not None else FilterResult(answer="no", percentageOfConfidence=50.0)

    detectors = {name: _or_neutral(results.get(name)) for name in FILTER_NAMES}

    # Calcular liveness (filtro 5 = score consolidado)
    attacks = [d for d in detectors.values() if d.answer == "yes"]
    if attacks:
        liveness = FilterResult(
            answer="yes",
            percentageOfConfidence=round(max(d.percentageOfConfidence for d in attacks), 1),
        )
    else:
        liveness = FilterResult(
            answer="no",
            percentageOfConfidence=round(min(d.percentageOfConfidence for d in detectors.values()), 1),
        )

    # Calcular veredicto consolidado
    is_authentic = liveness.answer == "no"
    # overall_confidence = grado de confianza de AUTENTICIDAD.
    # Si liveness dice "yes" (= ataque detectado), la confianza de
    # autenticidad es BAJA (= 100 - confianza del ataque).
    if is_authentic:
        overall_confidence = liveness.percentageOfConfidence
    else:
        overall_confidence = round(100.0 - liveness.percentageOfConfidence, 1)

    if overall_confidence >= 80:
        risk = "low"
    elif overall_confidence >= 60:
        risk = "medium"
    else:
        risk = "high"

    fraud_filter = next((n for n, d in detectors.items() if d.answer == "yes"), None)

    return {
        "file": str(image_path),
        "status": "ok",
        "verdict": {
            "is_authentic": is_authentic,
            "overall_confidence": overall_confidence,
            "risk_level": risk,
            "fraud_type": fraud_filter,
        },
        "filters": {
            "screen_capture":        detectors["screen_capture"].model_dump(),
            "printed_paper":         detectors["printed_paper"].model_dump(),
            "superimposed_elements": detectors["superimposed_elements"].model_dump(),
            "ai_altered":            detectors["ai_altered"].model_dump(),
            "liveness":              liveness.model_dump(),
        },
        "elapsed_ms": int((time.time() - t0) * 1000),
    }


# ─────────────────────────────────────────────────────────────────────
# Recolección de archivos
# ─────────────────────────────────────────────────────────────────────
def collect_files(target: Path, recursive: bool) -> list[Path]:
    if target.is_file():
        if target.suffix.lower() not in SUPPORTED_EXTS:
            print(RED(f"⚠ Archivo no soportado: {target.suffix}"))
            print(f"  Formatos válidos: {', '.join(sorted(SUPPORTED_EXTS))}")
            sys.exit(1)
        return [target]

    if not target.is_dir():
        print(RED(f"✗ La ruta no existe: {target}"))
        sys.exit(1)

    pattern = "**/*" if recursive else "*"
    files = [
        p for p in target.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]
    return sorted(files)


# ─────────────────────────────────────────────────────────────────────
# Renderizado de un resultado
# ─────────────────────────────────────────────────────────────────────
def print_result(result: dict, show_filename: bool = True) -> None:
    name = Path(result["file"]).name

    if result["status"] == "error":
        print(f"  {RED('✗')} {BOLD(name)}: {result['error']}")
        return

    v = result["verdict"]
    is_auth = v["is_authentic"]
    conf = v["overall_confidence"]
    risk = v["risk_level"]
    fraud = v["fraud_type"]

    icon = GREEN("✓ AUTÉNTICA") if is_auth else RED("✗ FRAUDE")
    risk_color = {"low": GREEN, "medium": YELLOW, "high": RED}[risk]
    elapsed = result["elapsed_ms"] / 1000

    if show_filename:
        print(f"\n  {BOLD(name)}  {DIM(f'({elapsed:.1f}s)')}")
    print(f"    {icon}  confianza={BOLD(f'{conf:.0f}%')}  riesgo={risk_color(risk.upper())}", end="")
    if fraud:
        print(f"  tipo={RED(fraud)}")
    else:
        print()

    print(f"    {DIM('────── Detalle de filtros ──────')}")
    f = result["filters"]
    label_map = {
        "screen_capture":        "1. Pantalla / cursor          ",
        "printed_paper":         "2. Impresión en papel         ",
        "superimposed_elements": "3. Stickers / superpuestos    ",
        "ai_altered":            "4. Generado por IA            ",
        "liveness":              "5. Autenticidad consolidada   ",
    }
    for key, label in label_map.items():
        ans = f[key]["answer"]
        c = f[key]["percentageOfConfidence"]
        marker = RED("yes") if ans == "yes" else GREEN("no ")
        print(f"      {label} {marker}  {c:5.1f}%")


def print_summary(results: list[dict]) -> None:
    valid = [r for r in results if r["status"] == "ok"]
    n_auth = sum(1 for r in valid if r["verdict"]["is_authentic"])
    n_fraud = len(valid) - n_auth
    n_err = len(results) - len(valid)

    avg_time = sum(r["elapsed_ms"] for r in results) / max(len(results), 1) / 1000

    print()
    print("═" * 70)
    print(BOLD("  RESUMEN"))
    print("═" * 70)
    print(f"  Total imágenes:       {len(results)}")
    print(f"  {GREEN('Auténticas:')}           {n_auth}")
    print(f"  {RED('Fraudes detectados:')}   {n_fraud}")
    if n_err:
        print(f"  {YELLOW('Errores:')}              {n_err}")
    print(f"  Tiempo promedio:      {avg_time:.1f}s/imagen")

    if n_fraud:
        print()
        print(BOLD("  Fraudes por tipo:"))
        from collections import Counter
        types = Counter(r["verdict"]["fraud_type"] or "OTHER"
                        for r in valid if not r["verdict"]["is_authentic"])
        for t, n in types.most_common():
            print(f"    {RED(t):30s} {n}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Verifica cédulas con el motor ADAMO ID.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("path", type=Path,
                        help="Ruta a una imagen o carpeta de imágenes")
    parser.add_argument("--workers", type=int, default=4,
                        help="Hilos paralelos para carpetas (def: 4)")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="Buscar imágenes en subcarpetas")
    parser.add_argument("--json", action="store_true",
                        help="Salida JSON en stdout (sin colores)")
    parser.add_argument("--output", "-o", type=Path,
                        help="Guardar resultados como JSON en este archivo")
    args = parser.parse_args()

    files = collect_files(args.path, args.recursive)
    if not files:
        print(YELLOW("⚠ No se encontraron imágenes en la ruta"))
        sys.exit(0)

    # Inicializar el motor
    if not args.json:
        print(DIM(f"Inicializando motor Gemini..."))
    engine = GeminiEngine()
    engine.initialize()
    if not engine.available:
        print(RED("✗ Gemini no disponible. Verifica que GEMINI_API_KEY esté en .env"))
        sys.exit(1)

    if not args.json:
        print(f"  {len(files)} imagen(es) a procesar  •  {args.workers} workers en paralelo")

    results: list[dict] = []
    t_start = time.time()

    if len(files) == 1:
        # Procesamiento simple
        result = verify_image(files[0], engine)
        results.append(result)
        if not args.json:
            print_result(result, show_filename=True)
    else:
        # Procesamiento paralelo
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            fut_map = {ex.submit(verify_image, fp, engine): fp for fp in files}
            done = 0
            for fut in as_completed(fut_map):
                fp = fut_map[fut]
                done += 1
                try:
                    result = fut.result()
                except Exception as exc:
                    result = {
                        "file": str(fp),
                        "status": "error",
                        "error": str(exc),
                        "elapsed_ms": 0,
                    }
                results.append(result)
                if not args.json:
                    print(DIM(f"  [{done:3d}/{len(files)}] "), end="")
                    print_result(result, show_filename=True)

    if not args.json:
        print_summary(results)
        print(DIM(f"\n  Tiempo total: {time.time() - t_start:.1f}s"))

    # Salida JSON
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))

    if args.output:
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\n{GREEN('✓')} Reporte guardado en: {args.output}")

    # Exit code: 0 = todo bien, 1 = al menos un fraude o error
    has_issues = any(
        r["status"] == "error" or not r["verdict"]["is_authentic"]
        for r in results
        if r["status"] == "ok" or r["status"] == "error"
    )
    sys.exit(1 if has_issues else 0)


if __name__ == "__main__":
    main()
