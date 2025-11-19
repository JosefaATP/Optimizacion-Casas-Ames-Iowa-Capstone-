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
        "n_estimators": 3758,
        "learning_rate": 0.03683247703306933,
        "max_depth": 4,
        "min_child_weight": 5.0,
        "gamma": 0.008345270600629286,
        "subsample": 0.6783424963567025,
        "colsample_bytree": 0.43051657642562313,
        "reg_lambda": 3.364253417277695,
        "reg_alpha": 0.052422719530854305,
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "n_jobs": -1,
        "random_state": 42,
    }


    # Si luego quieres restricciones monotónicas, defínelas aquí
    monotone: Optional[dict] = None
