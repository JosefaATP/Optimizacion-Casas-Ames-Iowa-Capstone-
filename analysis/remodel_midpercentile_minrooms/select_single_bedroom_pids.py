#!/usr/bin/env python3
"""
Selecciona una casa por barrio (idealmente con 1 dormitorio sobre rasante)
lo más cercana posible a percentiles objetivo de precio y escribe un CSV
para reutilizar en los análisis de sensibilidad.

Uso recomendado:
  python3 analysis/remodel_midpercentile_minrooms/select_single_bedroom_pids.py \
      --basecsv data/processed/base_completa_sin_nulos.csv \
      --outcsv analysis/remodel_midpercentile_minrooms/pids_single_bedroom.csv \
      --max-bedrooms 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def pick_row(sub: pd.DataFrame, price_col: str, targets: List[float]) -> tuple[pd.Series, float]:
    """Devuelve la fila cuyo precio está más cerca de alguno de los percentiles objetivo."""
    for pct in targets:
        q = sub[price_col].quantile(pct)
        idx = (sub[price_col] - q).abs().idxmin()
        if pd.isna(idx):
            continue
        row = sub.loc[idx]
        return row, pct
    # Fallback: simplemente el más cercano al precio medio
    q = sub[price_col].mean()
    idx = (sub[price_col] - q).abs().idxmin()
    return sub.loc[idx], 0.5


def build_catalog(df: pd.DataFrame, max_bedrooms: float | None, targets: List[float]) -> pd.DataFrame:
    rows = []
    price_col = "SalePrice_Present"
    for nb, sub in df.groupby("Neighborhood"):
        sub_nb = sub.copy()
        sub_nb["bed"] = pd.to_numeric(sub_nb.get("Bedroom AbvGr"), errors="coerce")
        filtered = sub_nb
        fallback = False
        if max_bedrooms is not None:
            filt = sub_nb["bed"] <= max_bedrooms
            if filt.any():
                filtered = sub_nb.loc[filt]
            else:
                fallback = True
                min_bed = sub_nb["bed"].min()
                filtered = sub_nb.loc[sub_nb["bed"] == min_bed]
        row, pct_used = pick_row(filtered, price_col, targets)
        price = float(row[price_col])
        base = sub_nb[price_col]
        actual_pct = (base <= price).mean()
        rows.append({
            "neighborhood": nb,
            "pid": int(row["PID"]),
            "base_price": price,
            "base_bedrooms": float(row["bed"]),
            "percentile_target": pct_used,
            "percentile_actual": actual_pct,
            "fallback_min_bed": fallback,
        })
    return pd.DataFrame(rows).sort_values("neighborhood")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Selecciona casas (1 dormitorio) por barrio.")
    ap.add_argument("--basecsv", type=Path, default=Path("data/processed/base_completa_sin_nulos.csv"))
    ap.add_argument("--outcsv", type=Path,
                    default=Path("analysis/remodel_midpercentile_minrooms/pids_single_bedroom.csv"))
    ap.add_argument("--outbase", type=Path, default=None,
                    help="Ruta opcional para escribir un CSV base con solo los PIDs seleccionados.")
    ap.add_argument("--max-bedrooms", type=float, default=1.0,
                    help="Máximo de dormitorios sobre rasante. Si un barrio no tiene, se toma el mínimo disponible.")
    ap.add_argument("--targets", type=float, nargs="+",
                    default=[0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8],
                    help="Percentiles objetivo para aproximar el precio.")
    return ap.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.basecsv)
    if "Neighborhood" not in df.columns or "PID" not in df.columns or "SalePrice_Present" not in df.columns:
        raise ValueError("El CSV debe incluir columns Neighborhood, PID y SalePrice_Present.")
    catalog = build_catalog(df, args.max_bedrooms, args.targets)
    args.outcsv.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(args.outcsv, index=False)
    print(f"[INFO] Guardado catálogo con {len(catalog)} barrios en {args.outcsv}")
    missing = catalog["fallback_min_bed"].sum()
    if missing:
        print(f"[WARN] {int(missing)} barrios no tenían casas <= {args.max_bedrooms} dormitorios; se tomó el mínimo disponible.")
    if args.outbase:
        subset = df[df["PID"].isin(catalog["pid"])]
        args.outbase.parent.mkdir(parents=True, exist_ok=True)
        subset.to_csv(args.outbase, index=False)
        print(f"[INFO] Guardado CSV base filtrado en {args.outbase}")


if __name__ == "__main__":
    main()
