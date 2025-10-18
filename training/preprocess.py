# src/preprocess.py
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline as SKPipeline

QUALITY_CANDIDATE_NAMES: List[str] = [
    "Kitchen Qual", "Exter Qual", "Exter Cond",
    "Bsmt Qual", "Bsmt Cond",
    "Heating QC",
    "Fireplace Qu",
    "Garage Qual", "Garage Cond",
    "Pool QC",
]

# Utilities ordinal
UTIL_ORDER = ["ELO", "NoSeWa", "NoSewr", "AllPub"]
UTIL_TO_ORD = {u:i for i,u in enumerate(UTIL_ORDER)}

# RoofStyle / RoofMatl (usamos las categorías del Ames “estándar”)
ROOF_STYLE_ORDER = ["Flat", "Gable", "Gambrel", "Hip", "Mansard", "Shed"]
ROOF_STYLE_TO_ORD = {u:i for i,u in enumerate(ROOF_STYLE_ORDER)}

ROOF_MATL_ORDER  = ["ClyTile", "CompShg", "Membran", "Metal", "Roll", "Tar&Grv", "WdShake", "WdShngl"]
ROOF_MATL_TO_ORD = {u:i for i,u in enumerate(ROOF_MATL_ORDER)}

def infer_feature_types(
    df: pd.DataFrame,
    target: str,
    drop_cols: List[str],
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    X = df.drop(columns=[target] + [c for c in drop_cols if c in df.columns], errors="ignore")
    if numeric_cols is None or categorical_cols is None:
        num_auto = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_auto = [c for c in X.columns if c not in num_auto]
        if numeric_cols is None:
            numeric_cols = num_auto
        if categorical_cols is None:
            categorical_cols = cat_auto
    return numeric_cols, categorical_cols

def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    quality_cols: List[str] | None = None,
    utilities_cols: List[str] | None = None,
    roof_cols: List[str] | None = None,
) -> ColumnTransformer:
    # TODO: el OHE ya estará horneado en el DataFrame, aquí solo pasamos numéricas
    quality_cols   = quality_cols or []
    utilities_cols = utilities_cols or []
    roof_cols      = roof_cols or []

    # tras get_dummies, todo es numérico; por seguridad consolida:
    all_numeric = sorted(list(set(numeric_cols + quality_cols + utilities_cols + roof_cols)))

    num_pipe = SKPipeline(steps=[("passthrough", "passthrough")])

    pre = ColumnTransformer(
        transformers=[("num", num_pipe, all_numeric)],
        remainder="drop",
        n_jobs=None,
    )
    return pre
