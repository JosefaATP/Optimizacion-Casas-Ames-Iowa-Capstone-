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

    # Parche extra: gurobi_ml 1.4 falla si base_score viene como "[5E-1]".
    try:
        import json, re
        import gurobi_ml.xgboost.xgboost_regressor as xr
        def _mip_model_safe(self, **kwargs):
            model = self.gp_model
            xgb_regressor = self.xgb_regressor

            _input = self._input
            output = self._output
            nex = _input.shape[0]
            timer = xr.AbstractPredictorConstr._ModelingTimer()
            outdim = output.shape[1]
            assert outdim == 1, "Output dimension of gradient boosting regressor should be 1"

            xgb_raw = json.loads(xgb_regressor.save_raw(raw_format="json"))
            bs_raw = xgb_raw["learner"]["learner_model_param"].get("base_score", 0.0)
            if isinstance(bs_raw, str) and "[" in bs_raw:
                m = re.match(r"\[\s*([0-9.eE+-]+)\s*\]", bs_raw)
                bs_raw = m.group(1) if m else bs_raw
            try:
                base_score = float(bs_raw)
            except Exception:
                base_score = 0.0

            booster_type = xgb_raw["learner"]["gradient_booster"]["name"]
            if booster_type != "gbtree":
                raise xr.NoModel(xgb_regressor, f"model not implemented for {booster_type}")
            trees = xgb_raw["learner"]["gradient_booster"]["model"]["trees"]
            n_estimators = len(trees)

            estimators = []
            if self._no_debug:
                kwargs["no_record"] = True

            tree_vars = model.addMVar(
                (nex, n_estimators, 1),
                lb=xr.GRB.INFINITY * -1,
                name=self._name_var("esimator"),
            )

            for i, tree in enumerate(trees):
                if self.verbose:
                    self._timer.timing(f"Estimator {i}")
                tree["threshold"] = (
                    xr.np.array(tree["split_conditions"], dtype=xr.np.float32) - self.epsilon
                )
                tree["children_left"] = xr.np.array(tree["left_children"])
                tree["children_right"] = xr.np.array(tree["right_children"])
                tree["feature"] = xr.np.array(tree["split_indices"])
                tree["value"] = tree["threshold"].reshape(-1, 1)
                tree["capacity"] = len(tree["split_conditions"])
                tree["n_features"] = int(tree["tree_param"]["num_feature"])

                estimators.append(
                    xr.AbstractTreeEstimator(
                        self.gp_model,
                        tree,
                        self.input,
                        tree_vars[:, i, :],
                        self.epsilon,
                        timer,
                        **kwargs,
                    )
                )

            self.estimators_ = estimators

            learning_rate = 1.0
            model.addConstr(output == learning_rate * tree_vars.sum(axis=1) + base_score)

        xr.XGBoostRegressorConstr._mip_model = _mip_model_safe
    except Exception:
        pass
