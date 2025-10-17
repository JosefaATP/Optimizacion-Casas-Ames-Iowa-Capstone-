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

UTIL_ORDER = ["ELO", "NoSeWa", "NoSewr", "AllPub"]
UTIL_TO_ORD = {u:i for i,u in enumerate(UTIL_ORDER)}

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
    utilities_cols: List[str] | None = None
) -> ColumnTransformer:
    # OJO: ya tratamos calidades como numéricas, así que no las metas a OHE
    quality_cols = quality_cols or []
    utilities_cols = utilities_cols or []

    # quítalas de las categóricas
    cat_ohe_cols = [c for c in categorical_cols if c not in quality_cols + utilities_cols]

    num_pipe = SKPipeline(steps=[("passthrough", "passthrough")])

    # OHE sólo para categóricas "normales"
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),     # ← aquí van también las Q/Cond ya ordinalizadas
            ("cat", ohe, cat_ohe_cols),
        ],
        remainder="drop",
        n_jobs=None,
    )
    return pre
