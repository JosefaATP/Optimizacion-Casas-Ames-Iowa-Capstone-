import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def infer_feature_types(df: pd.DataFrame, target: str, drop_cols: List[str],
                        numeric_cols: Optional[List[str]] = None,
                        categorical_cols: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
    X = df.drop(columns=[target] + [c for c in drop_cols if c in df.columns], errors="ignore")
    if numeric_cols is None or categorical_cols is None:
        num_auto = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_auto = [c for c in X.columns if c not in num_auto]
        if numeric_cols is None: numeric_cols = num_auto
        if categorical_cols is None: categorical_cols = cat_auto
    return numeric_cols, categorical_cols

def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ]
    )
    return pre
