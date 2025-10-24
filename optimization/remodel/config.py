from pathlib import Path
from pydantic import BaseModel

class Paths(BaseModel):
    # ruta de ESTE archivo
    file_path: Path = Path(__file__).resolve()
    # <repo>/optimization/remodel/config.py  ->  subimos 2 niveles al repo
    repo_root: Path = file_path.parents[2]  # .../Optimizacion-Casas-Ames-Iowa-Capstone-/
    data_dir: Path = repo_root / "data"
    # carpeta donde esta TU modelo elegido
    model_dir: Path = repo_root / "models" / "xgb" / "ordinal_p2_1800_ELEGIDO13"
    xgb_model_file: Path = model_dir / "model_xgb.joblib"
    # CSV por defecto (puedes sobreescribir por CLI si quieres)
    base_csv: Path = data_dir / "processed"/"base_completa_sin_nulos.csv"

class Params(BaseModel):
    mip_gap: float = 0.001
    time_limit: int = 300
    log_to_console: int = 1

PATHS = Paths()
PARAMS = Params()
