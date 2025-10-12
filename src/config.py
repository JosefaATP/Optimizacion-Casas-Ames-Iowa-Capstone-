from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Config:
    # nombre de la variable objetivo
    target: str = "precio"

    # columnas que NO quieres usar
    drop_cols: List[str] = field(default_factory=lambda: [
    "PID",
    "Order",           # <- evita fuga
    "SalePrice",        # <- muy importante, evita fuga
    # si tienes otras columnas derivadas del precio, agregalas aqui
    "\ufeffOrder",      # limpia la columna con BOM si existe
])
    


    # columnas numericas conocidas, si lo dejas vacio el script las infiere
    numeric_cols: Optional[List[str]] = None

    # columnas categoricas conocidas, si lo dejas vacio el script las infiere
    categorical_cols: Optional[List[str]] = None


    random_state: int = 42
    test_size: float = 0.2

    # hiperparametros iniciales de xgboost
    xgb_params = {
        "n_estimators": 1800,
        "learning_rate": 0.025,
        "max_depth": 5,           # árboles más bajos → menos varianza
        "min_child_weight": 10,   # todavía más conservador
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_lambda": 2.0,
        "reg_alpha": 0.0,
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "n_jobs": -1,
        "random_state": 42,
        }


    # si mas adelante quieres restricciones monotoniacas, se puede mapear aqui
    # ejemplo: {"m2": +1, "antiguedad": -1}
    monotone: Optional[dict] = None
