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
        "n_estimators": 2484,
      "learning_rate": 0.04999658913749818,
      "max_depth": 3,
      "min_child_weight": 7.0,
      "gamma": 0.0,
      "subsample": 0.3881362871329327,
      "colsample_bytree": 1.0,
      "reg_lambda": 5.0,
      "reg_alpha": 0.0,
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "n_jobs": -1,
        "random_state": 42,
    }


    # Si luego quieres restricciones monotónicas, defínelas aquí
    monotone: Optional[dict] = None
