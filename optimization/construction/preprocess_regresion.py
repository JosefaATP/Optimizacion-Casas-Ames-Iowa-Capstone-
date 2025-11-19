# -*- coding: utf-8 -*-
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .config import PATHS


REG_FEATURE_COLUMNS: List[str] = [
    # numeric/ordinal
    "Lot Frontage", "Lot Area", "Year Built", "Year Remod/Add", "Total Bsmt SF",
    "Low Qual Fin SF", "Gr Liv Area", "Bsmt Full Bath", "Bsmt Half Bath",
    "Full Bath", "Half Bath", "Bedroom AbvGr", "Kitchen AbvGr", "Fireplaces",
    "Garage Cars", "Wood Deck SF", "TotalPorch", "Overall Qual", "Overall Cond",
    "Exter Cond", "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1",
    "BsmtFin Type 2", "Heating QC", "Kitchen Qual", "Fireplace Qu",
    "Garage Finish", "Garage Qual", "MS SubClass",
    # categóricas
    "MS Zoning", "Lot Shape", "Lot Config", "Neighborhood", "Condition 1",
    "Roof Style", "Exterior 1st", "Foundation", "Central Air",
    "Garage Type", "Sale Type", "Sale Condition",
    # simplificadas binarias
    "Alley_simplificado", "Land_Contour_simplificado",
    "Land_Slope_simplificado", "Roof_Matl_simplificado",
    "Electrical_simplificado", "FunctioNo aplical_simplificado",
    "Paved_Drive_simplificado", "PoolQC_simplificado",
    "Fence_simplificado", "Misc_Feature_simplificado",
]


RAW_FOR_SIMPLIFIED = {
    "Alley_simplificado": "Alley",
    "Land_Contour_simplificado": "Land Contour",
    "Land_Slope_simplificado": "Land Slope",
    "Roof_Matl_simplificado": "Roof Matl",
    "Electrical_simplificado": "Electrical",
    "FunctioNo aplical_simplificado": "FunctioNo aplical",
    "Paved_Drive_simplificado": "Paved Drive",
    "PoolQC_simplificado": "Pool QC",
    "Fence_simplificado": "Fence",
    "Misc_Feature_simplificado": "Misc Feature",
}

PORCH_PARTS = ["Wood Deck SF", "Open Porch SF", "Enclosed Porch", "3Ssn Porch", "Screen Porch"]


def _norm(token: object) -> str:
    return "".join(ch for ch in str(token).lower() if ch.isalnum())


@lru_cache(maxsize=4)
def load_regression_reference_df(path: str | None = None) -> pd.DataFrame:
    csv_path = path or str(PATHS.repo_root / "data" / "raw" / "df_final_regresion.csv")
    df = pd.read_csv(csv_path)
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
    return df


def _get_value_from_sources(col: str, X_df: pd.DataFrame | None, base_row: pd.Series | None):
    if X_df is not None and col in X_df.columns:
        try:
            return X_df.loc[0, col]
        except Exception:
            pass
    if base_row is not None and col in base_row:
        return base_row.get(col)
    return None


def _fallback_from_df(col: str, ref_df: pd.DataFrame | None):
    if ref_df is None or col not in ref_df.columns:
        return None
    try:
        mode = ref_df[col].mode(dropna=True)
        if not mode.empty:
            return mode.iloc[0]
    except Exception:
        pass
    try:
        first = ref_df[col].dropna().iloc[0]
        return first
    except Exception:
        return None


def _canonicalize_category(value: object, column: str, ref_df: pd.DataFrame | None):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return _fallback_from_df(column, ref_df)
    if ref_df is None or column not in ref_df.columns:
        return value
    uniq = ref_df[column].dropna().astype(str).unique()
    mapper = {_norm(u): u for u in uniq}
    token = _norm(value)
    if token in mapper:
        return mapper[token]
    fallback = _fallback_from_df(column, ref_df)
    return fallback if fallback is not None else value


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(pd.to_numeric(value, errors="coerce"))
    except Exception:
        out = float("nan")
    if not np.isfinite(out):
        return float(default)
    return float(out)


def _compute_total_porch(X_df: pd.DataFrame | None, base_row: pd.Series | None) -> float:
    total = 0.0
    for col in PORCH_PARTS:
        val = _get_value_from_sources(col, X_df, base_row)
        if val is None:
            continue
        total += _as_float(val, 0.0)
    return float(total)


def _compute_simplified(name: str, raw_value: object) -> float:
    norm = _norm(raw_value) if raw_value is not None else ""
    if name in ("Alley_simplificado", "PoolQC_simplificado", "Fence_simplificado", "Misc_Feature_simplificado"):
        return 0.0 if norm in ("", "noaplica", "none", "nan") else 1.0
    if name == "Land_Contour_simplificado":
        return 1.0 if norm == "lvl" else 0.0
    if name == "Land_Slope_simplificado":
        return 1.0 if norm == "gtl" else 0.0
    if name == "Roof_Matl_simplificado":
        return 1.0 if norm == "compshg" else 0.0
    if name == "Electrical_simplificado":
        return 1.0 if norm == "sbrkr" else 0.0
    if name == "FunctioNo aplical_simplificado":
        return 1.0 if norm == "typ" else 0.0
    if name == "Paved_Drive_simplificado":
        return 1.0 if norm in ("y", "yes") else 0.0
    return 0.0


def prepare_regression_input(
    X_df: pd.DataFrame | None,
    base_row: pd.Series | None,
    feature_names: Iterable[str] | None,
    ref_df: pd.DataFrame | None,
) -> pd.DataFrame:
    cols = list(feature_names) if feature_names else list(REG_FEATURE_COLUMNS)
    features: Dict[str, object] = {}
    for col in cols:
        if col == "PID":
            pid_val = None
            if base_row is not None and "PID" in base_row:
                pid_val = base_row.get("PID")
            features[col] = _as_float(pid_val if pid_val is not None else -1.0, -1.0)
            continue
        if col in ("SalePrice", "SalePrice_Present"):
            base_val = None
            if base_row is not None and col in base_row:
                base_val = base_row.get(col)
            features[col] = _as_float(base_val if base_val is not None else 0.0, 0.0)
            continue
        if col == "TotalPorch":
            features[col] = _compute_total_porch(X_df, base_row)
            continue
        if col in RAW_FOR_SIMPLIFIED:
            source_col = RAW_FOR_SIMPLIFIED[col]
            raw_val = _get_value_from_sources(source_col, X_df, base_row)
            if raw_val is None:
                raw_val = _fallback_from_df(source_col, ref_df)
            features[col] = _compute_simplified(col, raw_val)
            continue
        val = _get_value_from_sources(col, X_df, base_row)
        if val is None:
            fallback = _fallback_from_df(col, ref_df)
            val = fallback
        if col in {"MS Zoning", "Lot Shape", "Lot Config", "Neighborhood", "Condition 1",
                   "Roof Style", "Exterior 1st", "Foundation", "Central Air",
                   "Garage Type", "Sale Type", "Sale Condition"}:
            features[col] = _canonicalize_category(val, col, ref_df)
        else:
            features[col] = _as_float(val, 0.0)

    cols = [c for c in cols if c not in ("SalePrice", "SalePrice_Present", "PID")]
    ordered = [features.get(c, 0.0) for c in cols]
    return pd.DataFrame([ordered], columns=cols)
