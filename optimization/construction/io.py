# optimization/construction/io.py
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from .config import PATHS

@dataclass
class BaseHouse:
    pid: int
    row: pd.Series

def load_base_df(base_csv: Path | None = None) -> pd.DataFrame:
    csv_path = Path(base_csv) if base_csv else PATHS.base_csv
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontro el CSV en {csv_path}\n"
            f"Tip: verifica que exista 'data/.../casas_completas_con_present.csv' dentro del repo,\n"
            f"o usa --basecsv RUTA/AL/CSV para pasar otra ruta."
        )
    df = pd.read_csv(csv_path, sep=None, engine="python")
    # limpiar BOM y espacios EXACTO como en tu train_xgb_log.py
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
    return df

    # autodetecta separador
    return pd.read_csv(csv_path, sep=None, engine="python")

def get_base_house(pid: int, base_csv: Path | None = None) -> BaseHouse:
    df = load_base_df(base_csv)
    row = df.loc[df["PID"] == pid]
    if row.empty:
        # ayuda rapida para encontrar un pid valido
        try:
            ejemplo = int(df["PID"].iloc[0])
        except Exception:
            ejemplo = None
        raise ValueError(f"PID {pid} no encontrado en base. "
                         f"{'Ejemplo de PID: ' + str(ejemplo) if ejemplo is not None else ''}")
    row = row.iloc[0]
    return BaseHouse(pid=int(row["PID"]), row=row)
