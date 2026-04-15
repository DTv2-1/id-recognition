"""
Script de prueba masiva — procesa todas las imágenes en una carpeta
y muestra los resultados de los 5 filtros en una tabla.

Uso:
    python scripts/test_batch.py --folder para-prueba-de-autenticidad
    python scripts/test_batch.py --folder para-prueba-de-autenticidad --api-key TU_KEY
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import time
from pathlib import Path

# Asegurar que el root del proyecto está en el path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

# Colores ANSI para terminal
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def image_to_b64(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    # Redimensionar si es muy grande (ahorra tokens y tiempo)
    max_side = 1024
    w, h = img.size
    if max(w, h) > max_side:
        ratio = max_side / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_verification(b64: str, api_key: str) -> dict:
    """Llama al pipeline completo con la imagen en base64."""
    os.environ["GEMINI_API_KEY"] = api_key

    # Import aquí para que el env var ya esté seteado
    from app.pipeline import VerificationPipeline
    from app.schemas import VerifyRequest

    pipeline = VerificationPipeline()
    pipeline.load_models()

    request = VerifyRequest(image=b64)
    response = pipeline.verify(request)
    return response.model_dump(mode="json")


def run_screen_capture_only(image_files: list[Path], api_key: str) -> None:
    """Prueba únicamente el filtro screen_capture con salida simplificada."""
    import json
    os.environ["GEMINI_API_KEY"] = api_key

    from app.filters.gemini_engine import GeminiEngine, _get_quality_context
    from PIL import Image as PILImage

    engine = GeminiEngine()
    engine.initialize()

    if not engine.available:
        print(f"{RED}Error: Gemini no disponible. Verifica GEMINI_API_KEY.{RESET}")
        sys.exit(1)

    print(f"\n{BOLD}PRUEBA FILTRO: screen_capture{RESET}  ({len(image_files)} imágenes)\n")
    print(f"  {BOLD}{'Archivo':<22}  {'Resultado':<10}  {'Confianza':>9}  {'Señales forenses'}{RESET}")
    print("  " + "─" * 95)

    detected = 0
    for img_path in image_files:
        try:
            img = PILImage.open(img_path).convert("RGB")
            # Métricas forenses (resumen en una línea)
            from app.filters.gemini_engine import _compute_forensic_signals
            sig = _compute_forensic_signals(img)
            metrics_str = (
                f"sat={sig['saturation']:.2f}  "
                f"B/R={sig['blue_ratio']:.2f}  "
                f"fft={sig['fft_peak_score']:.3f}  "
                f"dct={sig['dct_double_score']:.3f}  "
                f"alias={sig['aliasing_score']:.3f}"
            )

            result = engine.analyze_filter(img, "screen_capture")
            if result is None:
                print(f"  {CYAN}{img_path.name:<22}{RESET}  {RED}ERROR (None){RESET}")
                continue

            is_screen = result.answer == "yes"
            if is_screen:
                detected += 1
            color  = RED if is_screen else GREEN
            symbol = "✗ PANTALLA" if is_screen else "✓ REAL    "

            print(
                f"  {CYAN}{img_path.name:<22}{RESET}"
                f"  {color}{symbol}{RESET}"
                f"  {result.percentageOfConfidence:>8.1f}%"
                f"  {metrics_str}"
            )
        except Exception as exc:
            print(f"  {CYAN}{img_path.name:<22}{RESET}  {RED}ERROR: {exc}{RESET}")

    total = len(image_files)
    print()
    print("  " + "─" * 95)
    pct = detected / total * 100 if total else 0
    color = GREEN if pct >= 90 else (YELLOW if pct >= 70 else RED)
    print(f"  {BOLD}Detectadas: {color}{detected}/{total} ({pct:.0f}%){RESET}")
    print()


def run_printed_paper_only(image_files: list[Path], api_key: str) -> None:
    """Prueba únicamente el filtro printed_paper.

    Reconoce dos categorías por nombre de archivo:
      *-hoja-de-papel.*  → fraude esperado  (debe detectar  → detected=true ✓)
      *-original.*       → genuino esperado (no debe detectar → detected=false ✓)
    """
    os.environ["GEMINI_API_KEY"] = api_key

    from app.filters.gemini_engine import GeminiEngine, _compute_printed_paper_signals, _image_resize
    from PIL import Image as PILImage

    engine = GeminiEngine()
    engine.initialize()

    if not engine.available:
        print(f"{RED}Error: Gemini no disponible. Verifica GEMINI_API_KEY.{RESET}")
        sys.exit(1)

    print(f"\n{BOLD}PRUEBA FILTRO: printed_paper{RESET}  ({len(image_files)} imágenes)\n")
    col_w = 30
    print(f"  {BOLD}{'Archivo':<{col_w}}  {'Esperado':<10}  {'Resultado':<10}  {'Conf':>6}  {'OK?':<5}  {'Señales forenses'}{RESET}")
    print("  " + "─" * 120)

    tp = fp = tn = fn = 0  # true-pos, false-pos, true-neg, false-neg

    for img_path in image_files:
        try:
            # Determinar ground-truth por nombre de archivo
            name_lower = img_path.name.lower()
            if "hoja-de-papel" in name_lower or "hoja_de_papel" in name_lower:
                ground_truth = True   # es fraude, debe detectarse
            elif "original" in name_lower:
                ground_truth = False  # es genuino, no debe detectarse
            else:
                ground_truth = None   # desconocido

            img = PILImage.open(img_path).convert("RGB")
            img_small = _image_resize(img)   # misma imagen que ve Gemini
            sig = _compute_printed_paper_signals(img_small)
            metrics_str = (
                f"dct={sig['dct_triple_score']:.2f} "
                f"B/R={sig['blue_ratio']:.2f} "
                f"gc={sig['glcm_contrast']:.2f} "
                f"lap={sig.get('laplacian_variance', 0):.0f} "
                f"meij={sig.get('meijering_score', 0):.4f} "
                f"wHH={sig.get('wavelet_detail_energy', 0):.4f} "
                f"entr={sig.get('haralick_entropy', 0):.2f} "
                f"ht={sig['halftone_peak_score']:.2f} "
                f"cmyk={sig['cmyk_gamut_score']:.2f}"
            )

            result = engine.analyze_filter(img, "printed_paper")
            if result is None:
                print(f"  {CYAN}{img_path.name:<{col_w}}{RESET}  {RED}ERROR (None){RESET}")
                continue

            is_paper = result.answer == "yes"

            # Calcular métricas de clasificación
            if ground_truth is True:
                expected_str = f"{RED}PAPEL{RESET}    "
                if is_paper:
                    tp += 1
                    ok_str = f"{GREEN}✓ TP{RESET}"
                else:
                    fn += 1
                    ok_str = f"{RED}✗ FN{RESET}"
            elif ground_truth is False:
                expected_str = f"{GREEN}ORIGINAL{RESET}  "
                if not is_paper:
                    tn += 1
                    ok_str = f"{GREEN}✓ TN{RESET}"
                else:
                    fp += 1
                    ok_str = f"{RED}✗ FP{RESET}"
            else:
                expected_str = f"{YELLOW}?{RESET}         "
                ok_str = f"{YELLOW}?{RESET}   "

            res_color  = RED if is_paper else GREEN
            res_symbol = "PAPEL    " if is_paper else "NO PAPEL "

            print(
                f"  {CYAN}{img_path.name:<{col_w}}{RESET}"
                f"  {expected_str}"
                f"  {res_color}{res_symbol}{RESET}"
                f"  {result.percentageOfConfidence:>5.1f}%"
                f"  {ok_str}"
                f"  {metrics_str}"
            )
        except Exception as exc:
            print(f"  {CYAN}{img_path.name:<{col_w}}{RESET}  {RED}ERROR: {exc}{RESET}")

    total = tp + fp + tn + fn
    correct = tp + tn
    accuracy = correct / total * 100 if total else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0

    print()
    print("  " + "─" * 120)
    print(f"  {BOLD}RESULTADOS{RESET}")
    print(f"  Correctas : {GREEN if accuracy >= 80 else RED}{correct}/{total} ({accuracy:.0f}%){RESET}")
    print(f"  TP (papel detectado)   : {GREEN}{tp}{RESET}   FN (papel perdido) : {RED}{fn}{RESET}")
    print(f"  TN (original correcto) : {GREEN}{tn}{RESET}   FP (falso positivo): {RED}{fp}{RESET}")
    print(f"  Precision : {precision:.0f}%   Recall : {recall:.0f}%")
    print()


def format_filter(name: str, result: dict) -> str:
    answer = result["answer"]
    conf   = result["percentageOfConfidence"]

    # Para liveness "yes" es BUENO, para el resto "yes" es MALO
    if name == "liveness":
        good = (answer == "yes")
    else:
        good = (answer == "no")

    color  = GREEN if good else RED
    symbol = "✓" if good else "✗"
    return f"{color}{symbol} {answer:3s} {conf:5.1f}%{RESET}"


def print_row(filename: str, data: dict) -> None:
    v = data["verdict"]
    f = data["filters"]

    authentic = v["is_authentic"]
    conf      = v["overall_confidence"]
    risk      = v["risk_level"]
    ms        = data["processing_time_ms"]

    verdict_str = f"{GREEN}{BOLD}AUTÉNTICO{RESET}" if authentic else f"{RED}{BOLD}RECHAZADO{RESET}"
    risk_color  = GREEN if risk == "low" else (YELLOW if risk == "medium" else RED)

    sc  = format_filter("screen_capture",        f["screen_capture"])
    pp  = format_filter("printed_paper",          f["printed_paper"])
    se  = format_filter("superimposed_elements",  f["superimposed_elements"])
    ai  = format_filter("ai_altered",             f["ai_altered"])
    liv = format_filter("liveness",               f["liveness"])

    print(
        f"  {CYAN}{filename:<18}{RESET}"
        f"  {verdict_str}"
        f"  {conf:5.1f}%  "
        f"  {risk_color}{risk:<6}{RESET}"
        f"  SC:{sc}"
        f"  PP:{pp}"
        f"  SE:{se}"
        f"  AI:{ai}"
        f"  LV:{liv}"
        f"  {ms:>5}ms"
    )


def print_header() -> None:
    print()
    print(f"  {BOLD}{'Archivo':<18}  {'Veredicto':<18}  {'Conf':>5}  {'Riesgo':<6}"
          f"  {'SC (pantalla)':<16}"
          f"  {'PP (papel)':<16}"
          f"  {'SE (falsif)':<16}"
          f"  {'AI (ia)':<16}"
          f"  {'LV (live)':<16}"
          f"  {'Tiempo':>7}{RESET}")
    print("  " + "─" * 175)


def print_summary(results: list[dict]) -> None:
    total     = len(results)
    authentic = sum(1 for r in results if r["verdict"]["is_authentic"])
    rejected  = total - authentic
    avg_conf  = sum(r["verdict"]["overall_confidence"] for r in results) / total if total else 0
    avg_ms    = sum(r["processing_time_ms"] for r in results) / total if total else 0

    print()
    print("  " + "─" * 175)
    print(f"  {BOLD}RESUMEN{RESET}")
    print(f"  Total procesadas : {total}")
    print(f"  {GREEN}Auténticas       : {authentic}{RESET}")
    print(f"  {RED}Rechazadas       : {rejected}{RESET}")
    print(f"  Confianza media  : {avg_conf:.1f}%")
    print(f"  Tiempo medio     : {avg_ms:.0f}ms / imagen")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba masiva ADAMO ID")
    parser.add_argument("--folder", default="para-prueba-de-autenticidad",
                        help="Carpeta con las imágenes")
    parser.add_argument("--api-key", default="",
                        help="GEMINI_API_KEY (o setear variable de entorno)")
    parser.add_argument("--pattern", default="*.png,*.jpg,*.jpeg,*.webp",
                        help="Patrones de archivo separados por coma")
    parser.add_argument("--filter", default="all", choices=["all", "screen_capture", "printed_paper"],
                        help="Filtro a probar: 'all' (pipeline completo), 'screen_capture' o 'printed_paper'")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print(f"{RED}Error: GEMINI_API_KEY no configurada.{RESET}")
        print(f"  Usa --api-key TU_KEY  o  export GEMINI_API_KEY=TU_KEY")
        sys.exit(1)

    # Carpeta por defecto según el filtro elegido
    folder_arg = args.folder
    if folder_arg == "para-prueba-de-autenticidad" and args.filter == "printed_paper":
        folder_arg = "filtro-hoja-de-papel"

    folder = Path(folder_arg)
    if not folder.is_absolute():
        folder = Path(__file__).resolve().parent.parent / folder

    if not folder.exists():
        print(f"{RED}Error: carpeta '{folder}' no existe.{RESET}")
        sys.exit(1)

    # Recopilar imágenes
    patterns = [p.strip() for p in args.pattern.split(",")]
    image_files: list[Path] = []
    for pat in patterns:
        image_files.extend(sorted(folder.glob(pat)))

    if not image_files:
        print(f"{YELLOW}No se encontraron imágenes en {folder}{RESET}")
        sys.exit(0)

    # ── Modo filtro único ─────────────────────────────────────────
    if args.filter == "screen_capture":
        run_screen_capture_only(image_files, api_key)
        return
    if args.filter == "printed_paper":
        run_printed_paper_only(image_files, api_key)
        return

    # ── Modo pipeline completo ────────────────────────────────────
    print(f"\n{BOLD}ADAMO ID — Prueba masiva de autenticidad{RESET}")
    print(f"  Carpeta : {folder}")
    print(f"  Imágenes: {len(image_files)}")
    print(f"  Engine  : Gemini Vision (primario) + fallback local")

    # Inicializar pipeline una sola vez
    os.environ["GEMINI_API_KEY"] = api_key
    from app.pipeline import VerificationPipeline
    from app.schemas import VerifyRequest

    pipeline = VerificationPipeline()
    pipeline.load_models()
    engine = "Gemini" if pipeline.gemini.available else "Local (fallback)"
    print(f"  Motor activo: {CYAN}{engine}{RESET}")

    print_header()

    all_results = []
    errors = []

    for img_path in image_files:
        try:
            b64 = image_to_b64(img_path)
            request = VerifyRequest(image=b64)
            response = pipeline.verify(request)
            data = response.model_dump(mode="json")
            all_results.append(data)
            print_row(img_path.name, data)
        except Exception as exc:
            errors.append((img_path.name, str(exc)))
            print(f"  {CYAN}{img_path.name:<18}{RESET}  {RED}ERROR: {exc}{RESET}")

    print_summary(all_results)

    if errors:
        print(f"  {RED}Errores ({len(errors)}):{RESET}")
        for name, err in errors:
            print(f"    {name}: {err}")
        print()


if __name__ == "__main__":
    main()
