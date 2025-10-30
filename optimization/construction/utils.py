# en alguna util (p.ej. optimization/remodel/utils.py)
def debug_budget_report(m, tag="DEBUG PRESUPUESTO"):
    try:
        budget_con = m.getConstrByName("BUDGET")
        if budget_con is None:
            print("[DBG] No encontré la restricción BUDGET")
            return
        lhs = m.getVarByName("cost_model").X
        rhs = m.getConstrByName("BUDGET").RHS
        slack = rhs - lhs
        print(f"\n===== {tag} =====")
        print(f"BUDGET declarado en modelo: {rhs:,.2f}")
        print(f"LHS (costos totales): {lhs:,.2f}")
        print(f"RHS (presupuesto):    {rhs:,.2f}")
        print(f"Slack:                {slack:,.2f}")
        yp = getattr(m, "_y_price_var", None)
        if yp is not None:
            print(f"[DBG] y_price (predicho): {yp.X:,.2f}")
        print(f"[DBG] base_price_val (desde modelo): {getattr(m, '_base_price_val', float('nan')):,.2f}")
        print(f"[DBG] lin_cost.getValue(): {getattr(m, '_lin_cost_expr', None).getValue():,.2f}")
    except Exception as e:
        print(f"[DBG] Error en debug_budget_report: {e}")

def _link_onehot_to_ordinal(m, x, prefix, ord_name, value_map):
    # asegura sum z = 1 y ord = sum(val_k * z_k)
    zs = []
    expr = gp.LinExpr(0.0)
    for label, val in value_map.items():
        v = x.get(f"{prefix}{label}")
        if v is not None:
            zs.append(v)
            expr += float(val) * v
    if zs:
        m.addConstr(gp.quicksum(zs) == 1, name=f"ONEHOT_{prefix.strip('_')}")
        if ord_name in x:
            m.addConstr(x[ord_name] == expr, name=f"ORD_{ord_name.replace(' ','_')}")
