# optimization/remodel/xgb_predictor.py
from pathlib import Path
import json, joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "xgb" / "completa_present_log_p2_1800_ELEGIDO"

class XGBPricePredictor:
    def __init__(self):
        self.pipe = joblib.load(MODEL_DIR / "model_xgb.joblib")   # tu pipeline completo
        self.meta = json.load(open(MODEL_DIR / "meta.json","r",encoding="utf-8"))
        self.target = self.meta["target"]
        self.drop_cols = [c for c in self.meta.get("drop_cols", []) if c]

    def predict_price(self, feats: dict) -> float:
        df = pd.DataFrame([feats])
        X = df.drop(columns=[self.target] + self.drop_cols, errors="ignore")
        # ojo: tu preprocesador se encarga de imputar+onehot, y si entrenaste en log, TTR ya hace expm1
        y = self.pipe.predict(X)
        return float(y[0])
