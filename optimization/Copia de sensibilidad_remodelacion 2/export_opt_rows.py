"""
Extrae todas las casas remodeladas (opt_row) de `detalles.jsonl` y guarda un CSV por caso OPTIMAL.

Uso:
  python3 optimization/sensibilidad_remodelacion/export_opt_rows.py \
      --detalles optimization/sensibilidad_remodelacion/detalles.jsonl \
      --outdir optimization/sensibilidad_remodelacion/opt_rows
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

# asegurar repo en path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detalles", type=Path, required=True, help="Ruta a detalles.jsonl")
    ap.add_argument("--outdir", type=Path, default=Path("optimization/sensibilidad_remodelacion/opt_rows"),
                    help="Carpeta destino para los CSV de casas remodeladas")
    return ap.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    count = 0

    with args.detalles.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            meta = rec.get("meta", {})
            if meta.get("status") != "OPTIMAL":
                continue
            opt_row = rec.get("extra", {}).get("opt_row")
            if not opt_row:
                continue
            pid = meta.get("pid")
            budget = meta.get("budget")
            qfloor = meta.get("quality_floor") or "none"
            fname = f"opt_row_pid{pid}_b{int(budget)}_q{qfloor}.csv"
            out_csv = args.outdir / fname
            pd.DataFrame([opt_row]).to_csv(out_csv, index=False)
            count += 1
    print(f"Exportados {count} CSV en {args.outdir}")


if __name__ == "__main__":
    main()
