# optimization/remodel/compat_xgboost.py
def patch_get_booster():
    """Parche: si XGBRegressor.get_booster() lanza NotFittedError,
    pero existe self._Booster, devuélvelo."""
    try:
        from xgboost import XGBRegressor
    except Exception:
        return  # si no hay xgboost, no hacemos nada

    # Mantén referencia al original
    _orig = XGBRegressor.get_booster

    def _safe_get_booster(self, *args, **kwargs):
        try:
            return _orig(self, *args, **kwargs)
        except Exception:
            # fallback: algunos pickles tienen el booster en _Booster
            if hasattr(self, "_Booster") and self._Booster is not None:
                return self._Booster
            raise

    # Reemplaza una sola vez
    if XGBRegressor.get_booster is not _safe_get_booster:
        XGBRegressor.get_booster = _safe_get_booster
