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
