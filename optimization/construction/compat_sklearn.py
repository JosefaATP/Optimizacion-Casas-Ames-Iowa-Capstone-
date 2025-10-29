
# optimization/remodel/compat_sklearn.py
import importlib

def ensure_check_feature_names():
    """
    - Provee _check_feature_names si falta (gurobi_ml lo usa con versiones viejas/nuevas).
    - Parchea check_is_fitted para que no falle con sklearn.Pipeline recién construido
      cuando sus pasos ya están fitted (caso nuestro: pre -> XGB).
    Debe llamarse ANTES de importar gurobi_ml.
    """
    v = importlib.import_module("sklearn.utils.validation")

    # 1) Shim de _check_feature_names (acepta kwargs como reset=False)
    if not hasattr(v, "_check_feature_names"):
        def _check_feature_names(estimator, input_features, *args, **kwargs):
            return input_features
        v.__dict__["_check_feature_names"] = _check_feature_names

    # 2) Bypass suave de check_is_fitted para Pipeline
    if hasattr(v, "check_is_fitted"):
        _orig = v.check_is_fitted

        def _patched_check_is_fitted(estimator, *args, **kwargs):
            try:
                from sklearn.pipeline import Pipeline as SKPipeline
                # Si es un Pipeline, asumimos fitted (sus pasos ya lo están)
                if isinstance(estimator, SKPipeline):
                    return None
            except Exception:
                pass
            # Resto de estimadores: comportamiento original
            return _orig(estimator, *args, **kwargs)

        v.__dict__["check_is_fitted"] = _patched_check_is_fitted
