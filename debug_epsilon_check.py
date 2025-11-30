#!/usr/bin/env python3
"""
Detecta en qué árboles el embed de Gurobi elige una hoja distinta al booster.
Usa el modelo de remodel, resuelve un caso y compara hoja-por-hoja.
"""
from __future__ import annotations

import json
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd

from optimization.remodel.io import get_base_house
from optimization.remodel import costs
from optimization.remodel.gurobi_model import build_mip_embed
from optimization.remodel.xgb_predictor import XGBBundle
from optimization.remodel.run_opt import rebuild_embed_input_df

PROJECT_ROOT = Path(__file__).parent


def _gather_leaves(node) -> list[tuple[list[tuple[int, float, bool]], float]]:
    """Recorre el árbol como en attach_to_gurobi: yes (izq) primero, luego no."""
    leaves: list[tuple[list[tuple[int, float, bool]], float]] = []

    def walk(nd, path):
        if "leaf" in nd:
            leaves.append((path, float(nd["leaf"])))
            return
        f_idx = int(str(nd["split"]).replace("f", ""))
        thr = float(nd["split_condition"])
        yes_id = nd.get("yes")
        no_id = nd.get("no")

        yes_child = None
        no_child = None
        for ch in nd.get("children", []):
            if ch.get("nodeid") == yes_id:
                yes_child = ch
            elif ch.get("nodeid") == no_id:
                no_child = ch

        if yes_child is not None:
            walk(yes_child, path + [(f_idx, thr, True)])
        if no_child is not None:
            walk(no_child, path + [(f_idx, thr, False)])

    walk(node, [])
    return leaves


def _leaf_from_values(node, x_vec: np.ndarray) -> tuple[int, float, list[tuple[int, float, bool]]]:
    """Devuelve (idx_leaf, value, path) caminando con la regla x<thr vs >=thr."""
    leaves = _gather_leaves(node)
    # Mapear path a índice
    path_to_idx = {tuple(p): i for i, (p, _) in enumerate(leaves)}

    cur = node
    path = []
    while True:
        if "leaf" in cur:
            idx = path_to_idx.get(tuple(path), -1)
            return idx, float(cur.get("leaf", 0.0)), path
        f_idx = int(str(cur.get("split", "0")).replace("f", ""))
        thr = float(cur.get("split_condition", 0.0))
        yes_id = cur.get("yes")
        no_id = cur.get("no")
        val = float(x_vec[f_idx]) if f_idx < len(x_vec) else 0.0
        go_left = val < thr  # ties to right, como XGBoost
        next_id = yes_id if go_left else no_id
        nxt = None
        for ch in cur.get("children", []):
            if ch.get("nodeid") == next_id:
                nxt = ch
                break
        if nxt is None:
            # falla segura
            return -1, 0.0
        path.append((f_idx, thr, go_left))
        cur = nxt


def analyze(pid: int = 526351010, budget: float = 60000.0, top_details: int = 10):
    base = get_base_house(pid)
    ct = costs.CostTables()
    bundle = XGBBundle()
    base_row = base.row if hasattr(base, "row") else base

    m: gp.Model = build_mip_embed(base_row=base_row, budget=budget, ct=ct, bundle=bundle)
    m.setParam("OutputFlag", 0)
    m.optimize()
    if m.Status != gp.GRB.OPTIMAL and m.Status != gp.GRB.TIME_LIMIT:
        print(f"[STATUS] MIP not solved (status={m.Status})")
        return

    X_sol = rebuild_embed_input_df(m, getattr(m, "_X_base_numeric", pd.DataFrame()))
    if hasattr(X_sol, "to_numpy"):
        x_vec = X_sol.to_numpy()[0]
    else:
        x_vec = np.array(X_sol.iloc[0].tolist())

    # Suma de hojas externa
    dumps = bundle._get_dumps_limited()
    booster_order = list(bundle.booster_feature_order())

    # Orden de x_vec ya es booster_order porque rebuild_embed_input_df fuerza columnas
    leaf_mismatches = []
    sum_ext = 0.0
    sum_mip = 0.0

    for t_idx, js in enumerate(dumps):
        node = json.loads(js)
        leaves = _gather_leaves(node)
        # hoja elegida externamente
        ext_idx, ext_val, ext_path = _leaf_from_values(node, x_vec)
        sum_ext += ext_val

        # hoja elegida en el MIP
        mip_idx = None
        mip_path = None
        for k in range(len(leaves)):
            v = m.getVarByName(f"t{t_idx}_leaf{k}")
            if v is not None and v.X > 0.5:
                mip_idx = k
                mip_path = leaves[k][0]
                break
        if mip_idx is not None:
            sum_mip += leaves[mip_idx][1]

        if mip_idx != ext_idx:
            leaf_mismatches.append((t_idx, ext_idx, mip_idx, ext_val, leaves[mip_idx][1] if mip_idx is not None else None))
            if len(leaf_mismatches) <= top_details:
                print(f"[MISMATCH] Tree {t_idx}: ext={ext_idx} ({ext_val:.6f}) vs mip={mip_idx} ({leaves[mip_idx][1] if mip_idx is not None else None})")
                # Dump paths para debug (nombre feature, valor, thr, dir)
                feat_names = list(booster_order)
                def _fmt_path(path):
                    parts = []
                    for f_idx, thr, is_left in path:
                        name = feat_names[f_idx] if f_idx < len(feat_names) else f"f{f_idx}"
                        val = x_vec[f_idx] if f_idx < len(x_vec) else float("nan")
                        dir_txt = "<" if is_left else ">="
                        parts.append(f"{name} ({val:.6f}) {dir_txt} {thr:.6f}")
                    return " | ".join(parts)
                print(f"    ext path: {_fmt_path(ext_path)}")
                if mip_path is not None:
                    print(f"    mip path: {_fmt_path(mip_path)}")

    print(f"\nTotal trees: {len(dumps)}")
    print(f"Leaf mismatches: {len(leaf_mismatches)}")
    print(f"Sum leaves external: {sum_ext:.6f}")
    print(f"Sum leaves MIP:      {sum_mip:.6f}")
    print(f"Delta (MIP-ext):     {sum_mip - sum_ext:.6f}")

    try:
        y_log_raw = float(m.getVarByName("y_log_raw").X)
        print(f"y_log_raw (MIP var): {y_log_raw:.6f} | diff vs sum_mip: {y_log_raw - sum_mip:.6f}")
    except Exception:
        pass


if __name__ == "__main__":
    analyze()
