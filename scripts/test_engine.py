"""
Prueba el pipeline REAL (GeminiEngine + forensic_features) sobre las 3 carpetas.

Diferencia con /tmp/probe_all.py: éste llama a GeminiEngine.analyze_all(),
que es el código que realmente corre en producción — incluyendo el bloque
forense prepend y el override de saturación.

Lógica de etiquetas (misma que probe_all.py):
  para-prueba-de-autenticidad:
      photo_*  → GENUINE
      test-*   → FRAUD (screen capture desde monitor/webcam)
  filtro_3:
      todo     → FRAUD (algo encima: sticker, papel, hoja)
  filtro-hoja-de-papel:
      *-original*  → GENUINE
      resto        → FRAUD (dni impreso en papel)

Uso:
    python scripts/test_engine.py
    python scripts/test_engine.py --workers 4  # parallelismo
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Asegurar que el raíz del proyecto está en el path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from PIL import Image

from app.filters.gemini_engine import GeminiEngine, FILTER_NAMES

# ─────────────────────────────────────────────────────────────────────
# Carpetas y regla de etiquetado
# ─────────────────────────────────────────────────────────────────────
FOLDERS = [
    ROOT / "para-prueba-de-autenticidad",
    ROOT / "filtro_3",
    ROOT / "filtro-hoja-de-papel",
]

# Tipo de fraude esperado por carpeta (para diagnóstico)
EXPECTED_FRAUD_TYPE = {
    "para-prueba-de-autenticidad": "SCREEN_CAPTURE",
    "filtro_3": "SUPERIMPOSED",
    "filtro-hoja-de-papel": "PRINTED_PAPER",
}

def expected_for(folder: str, fname: str) -> str:
    base = os.path.basename(fname).lower()
    folder_name = os.path.basename(folder)
    if folder_name == "para-prueba-de-autenticidad":
        return "GENUINE" if base.startswith("photo_") else "FRAUD"
    if folder_name == "filtro_3":
        return "FRAUD"
    if folder_name == "filtro-hoja-de-papel":
        return "GENUINE" if "original" in base else "FRAUD"
    return "UNKNOWN"


def collect_images() -> list[tuple[str, str, str]]:
    """Retorna lista de (folder, filepath, expected_verdict)."""
    jobs = []
    for folder in FOLDERS:
        exts = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"]
        files: list[str] = []
        for ext in exts:
            files.extend(glob.glob(str(folder / ext)))
        files = sorted(set(files))
        for fp in files:
            exp = expected_for(str(folder), fp)
            if exp != "UNKNOWN":
                jobs.append((str(folder), fp, exp))
    return jobs


# ─────────────────────────────────────────────────────────────────────
# Llamada al engine real
# ─────────────────────────────────────────────────────────────────────

def probe_one(fp: str, engine: GeminiEngine) -> dict:
    """Llama GeminiEngine.analyze_all() y extrae veredicto consolidado."""

    image = Image.open(fp).convert("RGB")
    t0 = time.time()
    results = engine.analyze_all(image)
    elapsed = time.time() - t0

    # Veredicto consolidado: FRAUD si algún filtro dijo "yes"
    fraud_filter = None
    confidence = 0.0
    for name in FILTER_NAMES:
        r = results.get(name)
        if r and r.answer == "yes":
            fraud_filter = name
            confidence = r.percentageOfConfidence
            break

    # Si ningún filtro dijo "yes" → GENUINE con la conf del primer "no"
    if fraud_filter is None:
        first = results.get(FILTER_NAMES[0])
        confidence = first.percentageOfConfidence if first else 50.0

    # Mapear nombre de filtro a fraud_type legible
    filter_to_type = {
        "screen_capture": "SCREEN_CAPTURE",
        "printed_paper": "PRINTED_PAPER",
        "superimposed_elements": "SUPERIMPOSED",
        "ai_altered": "AI_GENERATED",
    }
    fraud_type = filter_to_type.get(fraud_filter) if fraud_filter else None
    verdict = "FRAUD" if fraud_filter else "GENUINE"

    return {
        "verdict": verdict,
        "fraud_type": fraud_type,
        "confidence": confidence,
        "elapsed": elapsed,
    }


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Test del engine real")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--folder", type=str, default=None,
                        help="Filtrar sólo una carpeta (ej: filtro_3)")
    args = parser.parse_args()

    jobs = collect_images()
    if args.folder:
        jobs = [(f, fp, e) for f, fp, e in jobs if args.folder in f]

    print(f"Total imágenes: {len(jobs)}  •  workers={args.workers}\n")

    engine = GeminiEngine()
    engine.initialize()
    if not engine.available:
        print("ERROR: Gemini engine no disponible. Verifica GEMINI_API_KEY en .env")
        sys.exit(1)

    results: dict[str, tuple[str, str, dict]] = {}
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut_map = {
            ex.submit(probe_one, fp, engine): (folder, fp, exp)
            for folder, fp, exp in jobs
        }
        done_n = 0
        for fut in as_completed(fut_map):
            folder, fp, exp = fut_map[fut]
            done_n += 1
            try:
                data = fut.result()
            except Exception as e:
                data = {"verdict": "ERR", "fraud_type": None, "confidence": 0,
                        "elapsed": 0, "error": str(e)[:200]}

            results[fp] = (folder, exp, data)
            verd = data["verdict"]
            ok = "✓" if verd == exp else "✗"
            ftype = data.get("fraud_type") or "-"
            print(
                f"  [{done_n:3d}/{len(jobs)}] {ok} "
                f"{os.path.basename(fp)[:44]:44s} "
                f"exp={exp:7s} pred={verd:7s} "
                f"{ftype:14s} conf={data['confidence']:.0f}% "
                f"({data['elapsed']:.1f}s)"
            )
            sys.stdout.flush()

    print(f"\nTiempo total: {time.time()-t0:.1f}s\n")

    # ── Tabla resumen ──────────────────────────────────────────────
    print("=" * 95)
    print("RESUMEN POR CARPETA  (engine real con forensic_features + override)")
    print("=" * 95)
    header = f"{'Carpeta':<40s} {'Total':>5} {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4} {'Acc%':>6}"
    print(header)
    print("-" * 95)

    g_tp = g_tn = g_fp = g_fn = 0
    for folder in FOLDERS:
        tp = tn = fp_ = fn = 0
        for path, (f, exp, data) in results.items():
            if f != str(folder):
                continue
            verd = data["verdict"]
            if exp == "FRAUD" and verd == "FRAUD":
                tp += 1
            elif exp == "GENUINE" and verd == "GENUINE":
                tn += 1
            elif exp == "GENUINE" and verd == "FRAUD":
                fp_ += 1
            elif exp == "FRAUD" and verd == "GENUINE":
                fn += 1
        tot = tp + tn + fp_ + fn
        acc = (tp + tn) / tot * 100 if tot else 0
        print(f"{os.path.basename(str(folder)):<40s} {tot:>5} {tp:>4} {tn:>4} {fp_:>4} {fn:>4} {acc:>5.1f}%")
        g_tp += tp; g_tn += tn; g_fp += fp_; g_fn += fn

    print("-" * 95)
    gtot = g_tp + g_tn + g_fp + g_fn
    gacc = (g_tp + g_tn) / gtot * 100 if gtot else 0
    print(f"{'GLOBAL':<40s} {gtot:>5} {g_tp:>4} {g_tn:>4} {g_fp:>4} {g_fn:>4} {gacc:>5.1f}%")
    print("\nTP=fraude detectado  TN=auténtico aprobado  FP=falso positivo  FN=fraude perdido")

    # ── Errores ────────────────────────────────────────────────────
    errs = [(p, d) for p, (f, e, d) in results.items() if d["verdict"] != e]
    if errs:
        print(f"\n── Errores ({len(errs)}) " + "─" * 60)
        for p, d in sorted(errs, key=lambda x: x[0]):
            _, exp, _ = results[p]
            folder_name = os.path.basename(os.path.dirname(p))
            exp_type = EXPECTED_FRAUD_TYPE.get(folder_name, "-") if exp == "FRAUD" else "GENUINE"
            got_type = d.get("fraud_type") or "GENUINE"
            flag = "⚠ wrong_type" if (
                exp == "FRAUD" and d["verdict"] == "FRAUD" and
                d.get("fraud_type") and got_type != exp_type
            ) else ""
            print(
                f"  {os.path.basename(p):44s}  "
                f"exp={exp_type:14s} → got={got_type:14s} conf={d['confidence']:.0f}%  {flag}"
            )


if __name__ == "__main__":
    main()
