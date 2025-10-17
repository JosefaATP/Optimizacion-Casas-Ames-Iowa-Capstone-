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

def infer_feature_types(
    df: pd.DataFrame,
    target: str,
    drop_cols: List[str],
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None
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
    quality_cols: Optional[List[str]] = None,  # <-- sigue existiendo por API, pero no usamos Ordinal
) -> ColumnTransformer:
    """
    Preprocesador SOLO con OneHotEncoder para TODAS las categóricas.
    (Nada de OrdinalEncoder: gurobi-ml no lo soporta.)
    Las restricciones de 'calidad' (Kitchen, etc.) se modelan en el MIP.
    """
    # Numérico: passthrough (tu dataset ya viene limpio)
    num_pipe = "passthrough"

    # Categóricas: OneHotEncoder (compat sklearn 1.1/1.2+)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=float)

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", ohe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre
