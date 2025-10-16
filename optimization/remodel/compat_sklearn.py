# optimization/remodel/compat_sklearn.py
import importlib

# optimization/remodel/compat_sklearn.py
import importlib

def ensure_check_feature_names():
    """
    Algunas versiones de gurobi_ml importan _check_feature_names desde sklearn.utils.validation.
    Si tu versión de sklearn no lo trae (o cambia la firma), lo definimos aquí ANTES de importar gurobi_ml.
    """
    v = importlib.import_module("sklearn.utils.validation")
    if not hasattr(v, "_check_feature_names"):
        def _check_feature_names(estimator, input_features, *args, **kwargs):
            # gurobi_ml puede pasar reset=False u otros kwargs; los absorbemos.
            # Para nuestro caso basta con devolver los nombres tal cual.
            return input_features
        v.__dict__["_check_feature_names"] = _check_feature_names
