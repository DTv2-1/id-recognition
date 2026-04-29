"""
Script de prueba masiva — procesa imágenes de una o varias carpetas y
muestra los resultados del pipeline (Gemini unificado) en una tabla.

Uso:
    # una sola carpeta:
    python scripts/test_batch.py --folder para-prueba-de-autenticidad

    # múltiples carpetas con etiqueta esperada:
    python scripts/test_batch.py --folders \\
        para-prueba-de-autenticidad:authentic \\
        filtro_3:rejected \\
        filtro-hoja-de-papel:rejected
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from PIL import Image

# Colores ANSI
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


# ─── Helpers ────────────────────────────────────────────────────────
def image_to_b64(path: Path, max_side: int = 1280) -> str:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        ratio = max_side / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _resolve_folder(folder_arg: str) -> Path:
    p = Path(folder_arg)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / p
    return p


def _format_filter(result: dict) -> str:
    answer = result["answer"]
    conf   = result["percentageOfConfidence"]
    good   = (answer == "no")  # "no" = no fraude = bueno
    color  = GREEN if good else RED
    symbol = "✓" if good else "✗"
    return f"{color}{symbol} {answer:3s} {conf:5.1f}%{RESET}"


# ─── Render ─────────────────────────────────────────────────────────
def print_header() -> None:
    print()
    print(f"  {BOLD}{'Archivo':<40}  {'Veredicto':<18}  {'Conf':>5}  {'Riesgo':<6}"
          f"  {'SC (pantalla)':<16}"
          f"  {'PP (papel)':<16}"
          f"  {'SE (falsif)':<16}"
          f"  {'AI (ia)':<16}"
          f"  {'LV (live)':<16}"
          f"  {'Tiempo':>7}{RESET}")
    print("  " + "─" * 195)


def print_row(filename: str, data: dict) -> None:
    v = data["verdict"]
    f = data["filters"]
    authentic = v["is_authentic"]
    conf      = v["overall_confidence"]
    risk      = v["risk_level"]
    ms        = data["processing_time_ms"]

    verdict_str = (f"{GREEN}{BOLD}AUTÉNTICO{RESET}"
                   if authentic else f"{RED}{BOLD}RECHAZADO{RESET}")
    risk_color  = GREEN if risk == "low" else (YELLOW if risk == "medium" else RED)

    sc  = _format_filter(f["screen_capture"])
    pp  = _format_filter(f["printed_paper"])
    se  = _format_filter(f["superimposed_elements"])
    ai  = _format_filter(f["ai_altered"])
    liv = _format_filter(f["liveness"])

    print(
        f"  {CYAN}{filename:<40}{RESET}"
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


def print_summary(results: list[dict]) -> None:
    """Resumen simple para modo --folder (sin etiquetas esperadas)."""
    total     = len(results)
    if not total:
        return
    authentic = sum(1 for r in results if r["verdict"]["is_authentic"])
    rejected  = total - authentic
    avg_conf  = sum(r["verdict"]["overall_confidence"] for r in results) / total
    avg_ms    = sum(r["processing_time_ms"] for r in results) / total
    print()
    print("  " + "─" * 195)
    print(f"  {BOLD}RESUMEN{RESET}")
    print(f"  Total procesadas : {total}")
    print(f"  {GREEN}Auténticas       : {authentic}{RESET}")
    print(f"  {RED}Rechazadas       : {rejected}{RESET}")
    print(f"  Confianza media  : {avg_conf:.1f}%")
    print(f"  Tiempo medio     : {avg_ms:.0f} ms / imagen\n")


def print_combined_summary(all_results: list[dict]) -> None:
    by_folder: dict[str, list[dict]] = defaultdict(list)
    for r in all_results:
        by_folder[r["_folder"]].append(r)

    print(f"\n  {BOLD}{'─'*90}{RESET}")
    print(f"  {BOLD}RESUMEN POR CARPETA{RESET}")
    print(f"  {'Carpeta':<35} {'Esperado':<12} {'Correcto':>8} {'Total':>6} {'Prec%':>6}"
          f"  {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4}")
    print(f"  {'─'*90}")

    g_correct = g_imgs = g_tp = g_tn = g_fp = g_fn = 0
    for folder_name, results in sorted(by_folder.items()):
        expected = results[0]["_expected"]
        tp = tn = fp = fn = 0
        for r in results:
            is_auth = r["verdict"]["is_authentic"]
            if expected == "authentic":
                if is_auth: tn += 1
                else:       fn += 1
            elif expected == "rejected":
                if not is_auth: tp += 1
                else:           fp += 1

        correct = tp + tn
        total   = len(results)
        pct     = correct / total * 100 if total else 0
        exp_str = ("AUTÉNTICO" if expected == "authentic"
                   else "RECHAZADO" if expected == "rejected" else "—")
        color   = GREEN if pct >= 80 else (YELLOW if pct >= 60 else RED)
        print(f"  {CYAN}{folder_name:<35}{RESET} {exp_str:<12} "
              f"{color}{correct:>8}/{total:<6}{RESET} {color}{pct:>5.0f}%{RESET}  "
              f"{GREEN}{tp:>4}{RESET} {GREEN}{tn:>4}{RESET} "
              f"{RED}{fp:>4}{RESET} {RED}{fn:>4}{RESET}")
        g_correct += correct; g_imgs += total
        g_tp += tp; g_tn += tn; g_fp += fp; g_fn += fn

    g_pct = g_correct / g_imgs * 100 if g_imgs else 0
    g_color = GREEN if g_pct >= 80 else (YELLOW if g_pct >= 60 else RED)
    print(f"  {'─'*90}")
    print(f"  {BOLD}{'GLOBAL':<35}{RESET} {'':<12} "
          f"{g_color}{g_correct:>8}/{g_imgs:<6}{RESET} {g_color}{g_pct:>5.0f}%{RESET}  "
          f"{GREEN}{g_tp:>4}{RESET} {GREEN}{g_tn:>4}{RESET} "
          f"{RED}{g_fp:>4}{RESET} {RED}{g_fn:>4}{RESET}")
    print("\n  TP=fraude detectado  TN=auténtico aprobado  "
          "FP=auténtico rechazado  FN=fraude no detectado\n")


# ─── Main ──────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba masiva ADAMO ID")
    parser.add_argument("--folder", default="",
                        help="Carpeta con las imágenes (modo carpeta única)")
    parser.add_argument("--folders", nargs="+", default=[],
                        metavar="CARPETA:ESPERADO",
                        help="Múltiples carpetas con etiqueta. "
                             "Ej: --folders carpeta_a:authentic carpeta_b:rejected")
    parser.add_argument("--api-key", default="",
                        help="GEMINI_API_KEY (o variable de entorno)")
    parser.add_argument("--pattern", default="*.png,*.jpg,*.jpeg,*.webp",
                        help="Patrones de archivo separados por coma")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print(f"{RED}Error: GEMINI_API_KEY no configurada.{RESET}")
        sys.exit(1)
    os.environ["GEMINI_API_KEY"] = api_key

    patterns = [p.strip() for p in args.pattern.split(",")]

    from app.pipeline import VerificationPipeline
    from app.schemas import VerifyRequest

    pipeline = VerificationPipeline()
    pipeline.load_models()
    engine = "Gemini" if pipeline.gemini.available else "Local (fallback)"

    def _process_folder(folder: Path, expected: str) -> list[dict]:
        img_files: list[Path] = []
        for pat in patterns:
            img_files.extend(sorted(folder.glob(pat)))
        exp_label = ("→ AUTÉNTICO" if expected == "authentic"
                     else "→ RECHAZADO" if expected == "rejected" else "")
        print(f"  {BOLD}{CYAN}{folder.name}{RESET}  ({len(img_files)} imágenes)  "
              f"{YELLOW}{exp_label}{RESET}")
        print_header()
        out: list[dict] = []
        for img_path in img_files:
            try:
                request = VerifyRequest(image=image_to_b64(img_path))
                response = pipeline.verify(request)
                data = response.model_dump(mode="json")
                data["_expected"] = expected
                data["_filename"] = img_path.name
                data["_folder"]   = folder.name
                out.append(data)
                print_row(img_path.name, data)
                sys.stdout.flush()
            except Exception as exc:
                print(f"  {CYAN}{img_path.name:<40}{RESET}  {RED}ERROR: {exc}{RESET}")
                sys.stdout.flush()
        print()
        return out

    # ── Modo multi-carpeta ──
    if args.folders:
        print(f"\n{BOLD}ADAMO ID — Prueba combinada multi-carpeta{RESET}")
        print(f"  Carpetas: {len(args.folders)}")
        print(f"  Motor   : {CYAN}{engine}{RESET}\n")

        all_results: list[dict] = []
        for entry in args.folders:
            folder_arg, expected = (entry.rsplit(":", 1) if ":" in entry
                                    else (entry, "auto"))
            folder = _resolve_folder(folder_arg)
            if not folder.exists():
                print(f"{RED}Error: carpeta '{folder}' no existe.{RESET}")
                continue
            all_results.extend(_process_folder(folder, expected))
        print_combined_summary(all_results)
        return

    # ── Modo carpeta única ──
    folder = _resolve_folder(args.folder or "para-prueba-de-autenticidad")
    if not folder.exists():
        print(f"{RED}Error: carpeta '{folder}' no existe.{RESET}")
        sys.exit(1)

    print(f"\n{BOLD}ADAMO ID — Prueba masiva{RESET}")
    print(f"  Carpeta : {folder}")
    print(f"  Motor   : {CYAN}{engine}{RESET}")
    results = _process_folder(folder, "auto")
    print_summary(results)


if __name__ == "__main__":
    main()
