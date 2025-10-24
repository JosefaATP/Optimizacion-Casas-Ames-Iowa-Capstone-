# training/preprocess.py
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SKPipeline

# === Calidades ===
# Ordinales (0..4), con “No aplica” = -1; se quedan como enteros
QUAL_ORD: List[str] = [
    "Kitchen Qual",
    "Exter Qual",
    "Exter Cond",
    "Heating QC",
    "Fireplace Qu",
    "Bsmt Qual",
    "Bsmt Cond",
    "Garage Qual",
    "Garage Cond",
    "Pool QC",
]

# Utilities ordinal
UTIL_ORDER = ["ELO", "NoSeWa", "NoSewr", "AllPub"]
UTIL_TO_ORD = {u: i for i, u in enumerate(UTIL_ORDER)}

# (referencia de techos si alguna vez vuelves a tratarlos ordinales)
ROOF_STYLE_ORDER = ["Flat", "Gable", "Gambrel", "Hip", "Mansard", "Shed"]
ROOF_MATL_ORDER  = ["ClyTile", "CompShg", "Membran", "Metal", "Roll", "Tar&Grv", "WdShake", "WdShngl"]

def infer_feature_types(
    df: pd.DataFrame,
    target: str,
    drop_cols: List[str],
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Tras get_dummies, casi todo será numérico. Esta función solo consolida listas.
    """
    X = df.drop(columns=[target] + [c for c in drop_cols if c in df.columns], errors="ignore")
    if numeric_cols is None or categorical_cols is None:
        num_auto = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_auto = [c for c in X.columns if c not in num_auto]
        if numeric_cols is None:
            numeric_cols = num_auto
        if categorical_cols is None:
            categorical_cols = cat_auto
    return numeric_cols, categorical_cols

def build_preprocessor(numeric_cols: List[str]) -> ColumnTransformer:
    # Nada de SKPipeline aquí
    return ColumnTransformer(
        transformers=[("num", "passthrough", numeric_cols)],
        remainder="drop",
        n_jobs=None,
    )
