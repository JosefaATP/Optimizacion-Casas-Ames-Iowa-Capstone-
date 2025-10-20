from typing import Dict, Any, List, Tuple

def _fmt_money(x: float) -> str:
    return f"${x:,.0f}"

def _fmt_float(x: float) -> str:
    return f"{x:,.1f}"

def render_presentable_summary(
    *,
    pid: Any,
    neighborhood: str,
    budget: float,
    model_name: str,
    mip_gap: float,
    time_sec: float,
    base_price: float,
    opt_price: float,
    total_cost: float,
    objective: float,
    slack: float,
    top_changes: List[Tuple[str, str, str]],  # [(attr, base_str, opt_str)]
    cost_lines: List[Tuple[str, float]] = None # optional [(varname, cost)]
) -> str:
    lines = []
    lines.append("")
    lines.append("============================================================")
    lines.append("               RESULTADOS DE LA OPTIMIZACIÓN")
    lines.append("============================================================")
    lines.append(f"📍 PID: {pid} – {neighborhood} | Presupuesto: {_fmt_money(budget)}")
    lines.append(f"🧮 Modelo: {model_name}")
    lines.append(f"⏱️ Tiempo total: {_fmt_float(time_sec)}s | MIP Gap: {mip_gap*100:.4f}%")
    lines.append("")
    lines.append("💰 Resumen Económico")
    lines.append(f"  Precio casa base:        {_fmt_money(base_price)}")
    lines.append(f"  Precio casa remodelada:  {_fmt_money(opt_price)}")
    lines.append(f"  Δ Precio:                {_fmt_money(opt_price - base_price)}")
    lines.append(f"  Costos totales:          {_fmt_money(total_cost)}")
    lines.append(f"  Valor objetivo (MIP):    {_fmt_money(objective)}   (≡ y_price - total_cost)")
    lines.append(f"  Utilidad vs. base:       {_fmt_money((opt_price - total_cost) - base_price)}")
    lines.append(f"  Slack presupuesto:       {_fmt_money(slack)}")
    lines.append("")
    if top_changes:
        lines.append("🏠 Cambios clave")
        for a, b, c in top_changes:
            lines.append(f"  - {a}: {b} → {c}")
        lines.append("")
    if cost_lines:
        lines.append("🔎 Desglose de costos (principales términos)")
        for nm, c in cost_lines:
            lines.append(f"   · {nm:<28} {_fmt_money(c)}")
        lines.append("")
    lines.append("============================================================")
    lines.append("            FIN RESULTADOS DE LA OPTIMIZACIÓN")
    lines.append("============================================================")
    return "\n".join(lines)
