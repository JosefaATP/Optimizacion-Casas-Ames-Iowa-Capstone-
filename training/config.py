from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Config:
    target: str = "SalePrice_Present"

    drop_cols: List[str] = field(default_factory=lambda: [
        "PID",
        "Order",
        "SalePrice",
        "\ufeffOrder",
    ])

    # Si se dejan en None, se inferirán tras get_dummies
    numeric_cols: Optional[List[str]] = None
    categorical_cols: Optional[List[str]] = None

    random_state: int = 42
    test_size: float = 0.2

    xgb_params = {
        "n_estimators": 2843,
        "learning_rate": 0.042345759919321546,
        "max_depth": 3,
        "min_child_weight": 4.0,
        "gamma": 0.050035425161215466,
        "subsample": 0.5205384818739047,
        "colsample_bytree": 0.5693415844980262,
        "reg_lambda": 3.827629507754387,
        "reg_alpha": 0.05963017089026609,
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "n_jobs": -1,
        "random_state": 42,
    }

    # Si luego quieres restricciones monotónicas, defínelas aquí
    monotone: Optional[dict] = None
