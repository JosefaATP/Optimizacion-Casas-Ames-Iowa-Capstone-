import pandas as pd
from pathlib import Path
import json
from collections import defaultdict, Counter
import os

# Permite sobreescribir rutas vía variables de entorno
RESUMEN = Path(os.getenv("SENSI_RESUMEN", "optimization/sensibilidad_remodelacion/resumen.csv"))
OUT_TXT = Path(os.getenv("SENSI_OUT_TXT", "optimization/sensibilidad_remodelacion/roi_summary.txt"))
DETALLES = Path(os.getenv("SENSI_DETALLES", "optimization/sensibilidad_remodelacion/detalles.jsonl"))
HEATMAP_DIR = Path(os.getenv("SENSI_OUT_DIR", "optimization/sensibilidad_remodelacion"))

# Clasificación de barrios (según costos/umbral compartido)
NEIGH_CATS = {
    "baja": {
        "MeadowV", "IDOTRR", "BrDale", "OldTown", "BrkSide", "Edwards",
        "SWISU", "Sawyer", "NPkVill", "Blueste",
    },
    "media": {
        "Landmrk", "No aplicames", "Mitchel", "SawyerW", "NWAmes",
        "Gilbert", "Greens", "Blmngtn", "CollgCr",
    },
    "alta": {
        "Crawfor", "ClearCr", "Somerst", "Timber", "Veenker",
        "GrnHill", "NridgHt", "StoneBr", "NoRidge",
    },
}

def safe_roi(row):
    c = row.get("cost", 0)
    g = row.get("net_gain", 0)
    try:
        c = float(c); g = float(g)
    except Exception:
        return 0.0
    return g / c if c else 0.0

if not RESUMEN.exists():
    raise SystemExit(f"No se encontró {RESUMEN}")

df = pd.read_csv(RESUMEN)
if df.empty:
    raise SystemExit("resumen.csv está vacío")

for col in ["neighborhood", "percentile_label", "net_gain", "cost", "budget_used_pct", "budget"]:
    if col not in df.columns:
        raise SystemExit(f"Falta columna {col} en resumen.csv")

df["roi_pct"] = df.apply(safe_roi, axis=1)
df["roi_pct_sign"] = df["roi_pct"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
# Uplift relativo sobre precio base si está disponible
if "base_price" in df.columns:
    df["uplift_pct_base"] = df.apply(
        lambda r: (r["net_gain"] / r["base_price"]) if r.get("base_price") not in (None, 0) else 0.0,
        axis=1,
    )

# Mapear categoría de barrio
cat_map = {}
for cat, names in NEIGH_CATS.items():
    for n in names:
        cat_map[n] = cat

df["neigh_cat"] = df["neighborhood"].map(cat_map).fillna("sin_categoria")

mpl_cfg = HEATMAP_DIR / ".mplconfig"
mpl_cfg.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cfg))
os.environ.setdefault("MPLBACKEND", "Agg")

# Agrupar por barrio y percentil
agg = df.groupby(["neighborhood", "percentile_label"]).agg(
    cases=("pid", "count"),
    avg_roi_pct=("roi_pct", "mean"),
    avg_net_gain=("net_gain", "mean"),
    avg_cost=("cost", "mean"),
    avg_budget_used_pct=("budget_used_pct", "mean"),
).reset_index()

# Agregado solo por percentil (sin separar barrio)
by_pct = df.groupby("percentile_label").agg(
    cases=("pid", "count"),
    avg_roi_pct=("roi_pct", "mean"),
    avg_net_gain=("net_gain", "mean"),
    avg_cost=("cost", "mean"),
    avg_budget_used_pct=("budget_used_pct", "mean"),
).reset_index()

# Agregado solo por presupuesto
by_budget = df.groupby("budget").agg(
    cases=("pid", "count"),
    avg_roi_pct=("roi_pct", "mean"),
    avg_net_gain=("net_gain", "mean"),
    avg_cost=("cost", "mean"),
    avg_budget_used_pct=("budget_used_pct", "mean"),
).reset_index()

# Agregado por categoría de barrio
by_cat = df.groupby("neigh_cat").agg(
    cases=("pid", "count"),
    avg_roi_pct=("roi_pct", "mean"),
    med_roi_pct=("roi_pct", "median"),
    std_roi_pct=("roi_pct", "std"),
    avg_net_gain=("net_gain", "mean"),
    avg_cost=("cost", "mean"),
    avg_budget_used_pct=("budget_used_pct", "mean"),
    pos_roi=("roi_pct_sign", lambda s: int((s > 0).sum())),
    neg_roi=("roi_pct_sign", lambda s: int((s < 0).sum())),
).reset_index()

# Agregado por categoría y presupuesto
by_cat_budget = df.groupby(["neigh_cat", "budget"]).agg(
    cases=("pid", "count"),
    avg_roi_pct=("roi_pct", "mean"),
    med_roi_pct=("roi_pct", "median"),
    avg_net_gain=("net_gain", "mean"),
    avg_cost=("cost", "mean"),
    avg_budget_used_pct=("budget_used_pct", "mean"),
    pos_roi=("roi_pct_sign", lambda s: int((s > 0).sum())),
    neg_roi=("roi_pct_sign", lambda s: int((s < 0).sum())),
).reset_index()

# Cambios más frecuentes por categoría/presupuesto/percentil (solo diferencias registradas en detalles.jsonl)
top_changes_cat = defaultdict(Counter)
top_changes_budget = defaultdict(Counter)
top_changes_pct = defaultdict(Counter)
# Acumuladores de métricas de áreas/ambientes base vs óptimo
numeric_fields = [
    "Full Bath", "Half Bath", "Bedroom AbvGr", "Kitchen AbvGr",
    "Garage Area", "1st Flr SF", "2nd Flr SF", "Gr Liv Area",
    "Bsmt Unf SF", "BsmtFin SF 1", "BsmtFin SF 2",
    "Wood Deck SF", "Open Porch SF", "Enclosed Porch", "Screen Porch", "3Ssn Porch",
]
area_records = []
if DETALLES.exists():
    with DETALLES.open() as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            meta = obj.get("meta", {})
            extra = obj.get("extra", {})
            nb = meta.get("neighborhood")
            cat = cat_map.get(nb, "sin_categoria")
            budget_val = meta.get("budget")
            pct_label = meta.get("percentile_label")
            for ch in extra.get("changes", []):
                col = ch.get("col")
                if not col:
                    continue
                top_changes_cat[cat][col] += 1
                if budget_val is not None:
                    top_changes_budget[str(int(float(budget_val)))] += Counter({col: 1})
                if pct_label is not None:
                    top_changes_pct[str(pct_label)] += Counter({col: 1})
            base_row = extra.get("base_row") or {}
            opt_row = extra.get("opt_row") or {}
            rec = {"neigh_cat": cat, "budget": budget_val, "percentile": pct_label}
            for f in numeric_fields:
                try:
                    bval = float(pd.to_numeric(base_row.get(f, None), errors="coerce"))
                except Exception:
                    bval = None
                try:
                    oval = float(pd.to_numeric(opt_row.get(f, None), errors="coerce"))
                except Exception:
                    oval = None
                if pd.notna(bval):
                    rec[f"base::{f}"] = bval
                if pd.notna(oval):
                    rec[f"opt::{f}"] = oval
                if pd.notna(bval) and pd.notna(oval):
                    rec[f"delta::{f}"] = oval - bval
            area_records.append(rec)

# Helper para heatmaps
def save_heatmap(counter_map: dict, index_order: list[str], title: str, filename: Path):
    try:
        import matplotlib
        matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.colors import LinearSegmentedColormap  # type: ignore
    except Exception as e:
        print(f"Saltando heatmap {title} (matplotlib no disponible: {e})")
        return
    # Unir todas las columnas
    all_cols = set()
    for c in counter_map.values():
        all_cols.update(c.keys())
    if not all_cols:
        return
    # ordenar columnas por frecuencia total desc
    totals = Counter()
    for c in counter_map.values():
        totals.update(c)
    cols_sorted = [c for c, _ in totals.most_common()]
    rows = index_order or list(counter_map.keys())
    data = []
    for r in rows:
        freq = counter_map.get(r, {})
        data.append([freq.get(c, 0) for c in cols_sorted])
    cmap = LinearSegmentedColormap.from_list("green_red", ["#0b6623", "#ff0000"])
    fig, ax = plt.subplots(figsize=(max(8, len(cols_sorted)*0.4), max(3, len(rows)*0.5)))
    im = ax.imshow(data, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(cols_sorted)))
    ax.set_xticklabels(cols_sorted, rotation=90, fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.7)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def save_area_heatmap(area_df: pd.DataFrame, group_col: str, title: str, filename: Path):
    try:
        import matplotlib
        matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
        from matplotlib.colors import LinearSegmentedColormap  # type: ignore
    except Exception as e:
        print(f"Saltando heatmap áreas {title} (matplotlib no disponible: {e})")
        return
    if area_df.empty:
        return
    rows = sorted(area_df[group_col].dropna().unique())
    if not rows:
        return
    # columnas: delta% = (opt-base)/base *100
    cols = numeric_fields
    data = []
    for r in rows:
        sub = area_df.loc[area_df[group_col] == r]
        row_vals = []
        for f in cols:
            b_col, o_col = f"base::{f}", f"opt::{f}"
            if b_col not in sub.columns or o_col not in sub.columns:
                row_vals.append(0.0)
                continue
            b = sub[b_col]
            o = sub[o_col]
            delta_pct = ((o - b) / b.replace(0, pd.NA)) * 100
            row_vals.append(delta_pct.mean(skipna=True))
        data.append(row_vals)
    data_arr = pd.DataFrame(data, index=rows, columns=cols)
    cmap = LinearSegmentedColormap.from_list("green_red", ["#0b6623", "#ff0000"])
    fig, ax = plt.subplots(figsize=(max(10, len(cols)*0.5), max(3, len(rows)*0.4)))
    im = ax.imshow(data_arr.values, aspect="auto", cmap=cmap, vmin=-50, vmax=50)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_title(title + " (Δ% vs base)")
    fig.colorbar(im, ax=ax, shrink=0.7)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def save_area_bars(area_df: pd.DataFrame, group_col: str, prefix: str):
    """Genera barras por grupo (mean delta por campo). Base se deja implícita; sólo se colorea el delta."""
    try:
        import matplotlib
        matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
        from matplotlib.colors import LinearSegmentedColormap  # type: ignore
    except Exception as e:
        print(f"Saltando barras áreas {group_col} (matplotlib no disponible: {e})")
        return
    cmap = LinearSegmentedColormap.from_list("green_red", ["#0b6623", "#ff0000"])
    groups = sorted(area_df[group_col].dropna().unique())
    for g in groups:
        sub = area_df.loc[area_df[group_col] == g]
        means = []
        for f in numeric_fields:
            d_col = f"delta::{f}"
            means.append(sub[d_col].mean() if d_col in sub else 0.0)
        x = np.arange(len(numeric_fields))
        colors = [cmap(0.2) if m < 0 else cmap(0.8) for m in means]
        fig, ax = plt.subplots(figsize=(max(10, len(numeric_fields)*0.5), 4))
        ax.bar(x, means, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(numeric_fields, rotation=90, fontsize=8)
        ax.set_title(f"{prefix} {g} (Δ absoluta vs base)")
        ax.axhline(0, color="gray", linewidth=0.8)
        out = HEATMAP_DIR / f"bars_{prefix}_{g}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=200)
        plt.close(fig)

lines = []
for _, row in agg.sort_values(["neighborhood", "percentile_label"]).iterrows():
    lines.append(
        f"{row['neighborhood']} | {row['percentile_label']}: ROI%={row['avg_roi_pct']*100:.2f}% | "
        f"Δ$={row['avg_net_gain']:,.0f} | Costo prom={row['avg_cost']:,.0f} | "
        f"Presupuesto usado={row['avg_budget_used_pct']*100:.1f}% | casos={int(row['cases'])}"
    )

lines.append("")
lines.append("--- ROI por percentil (global) ---")
for _, row in by_pct.sort_values("percentile_label").iterrows():
    lines.append(
        f"{row['percentile_label']}: ROI%={row['avg_roi_pct']*100:.2f}% | "
        f"Δ$={row['avg_net_gain']:,.0f} | Costo prom={row['avg_cost']:,.0f} | "
        f"Presupuesto usado={row['avg_budget_used_pct']*100:.1f}% | casos={int(row['cases'])}"
    )

lines.append("")
lines.append("--- ROI por presupuesto (global) ---")
for _, row in by_budget.sort_values("budget").iterrows():
    lines.append(
        f"{int(row['budget'])}: ROI%={row['avg_roi_pct']*100:.2f}% | "
        f"Δ$={row['avg_net_gain']:,.0f} | Costo prom={row['avg_cost']:,.0f} | "
        f"Presupuesto usado={row['avg_budget_used_pct']*100:.1f}% | casos={int(row['cases'])}"
    )

lines.append("")
lines.append("--- ROI por categoría de barrio ---")
for _, row in by_cat.sort_values("neigh_cat").iterrows():
    lines.append(
        f"{row['neigh_cat']}: ROI%={row['avg_roi_pct']*100:.2f}% (med={row['med_roi_pct']*100:.2f}%, std={row['std_roi_pct']*100:.2f}%) | "
        f"Δ$={row['avg_net_gain']:,.0f} | Costo prom={row['avg_cost']:,.0f} | "
        f"Presupuesto usado={row['avg_budget_used_pct']*100:.1f}% | casos={int(row['cases'])} | "
        f"ROI +:{int(row['pos_roi'])} / ROI -:{int(row['neg_roi'])}"
    )

lines.append("")
lines.append("--- Cambios más frecuentes por categoría (solo diferencias) ---")
for cat in ["baja", "media", "alta"]:
    freq = top_changes_cat.get(cat, {})
    top5 = sorted(freq.items(), key=lambda t: t[1], reverse=True)[:5]
    pretty = ", ".join([f"{col} ({cnt})" for col, cnt in top5]) if top5 else "N/A"
    lines.append(f"{cat}: {pretty}")

lines.append("")
lines.append("--- Cambios menos frecuentes por categoría (solo diferencias) ---")
for cat in ["baja", "media", "alta"]:
    freq = {k: v for k, v in top_changes_cat.get(cat, {}).items() if v > 0}
    low5 = sorted(freq.items(), key=lambda t: t[1])[:5]
    pretty = ", ".join([f"{col} ({cnt})" for col, cnt in low5]) if low5 else "N/A"
    lines.append(f"{cat}: {pretty}")

lines.append("")
lines.append("--- ROI por categoría de barrio y presupuesto ---")
for _, row in by_cat_budget.sort_values(["neigh_cat", "budget"]).iterrows():
    lines.append(
        f"{row['neigh_cat']} | {int(row['budget'])}: ROI%={row['avg_roi_pct']*100:.2f}% (med={row['med_roi_pct']*100:.2f}%) | "
        f"Δ$={row['avg_net_gain']:,.0f} | Costo prom={row['avg_cost']:,.0f} | "
        f"Presupuesto usado={row['avg_budget_used_pct']*100:.1f}% | casos={int(row['cases'])} | "
        f"ROI +:{int(row['pos_roi'])} / ROI -:{int(row['neg_roi'])}"
    )

# Heatmaps de cambios
save_heatmap(top_changes_cat, ["baja", "media", "alta"], "Cambios por categoría de barrio", HEATMAP_DIR / "heatmap_changes_by_cat.png")
budget_order = sorted({b for b in top_changes_budget.keys()}, key=lambda x: float(x))
save_heatmap(top_changes_budget, budget_order, "Cambios por presupuesto", HEATMAP_DIR / "heatmap_changes_by_budget.png")
pct_order = sorted({p for p in top_changes_pct.keys()})
save_heatmap(top_changes_pct, pct_order, "Cambios por percentil", HEATMAP_DIR / "heatmap_changes_by_percentil.png")

lines.append("")
lines.append("--- Cambios más frecuentes por presupuesto ---")
for b in budget_order:
    freq = top_changes_budget.get(b, {})
    top5 = sorted(freq.items(), key=lambda t: t[1], reverse=True)[:5]
    pretty = ", ".join([f"{col} ({cnt})" for col, cnt in top5]) if top5 else "N/A"
    lines.append(f"{b}: {pretty}")

lines.append("")
lines.append("--- Cambios menos frecuentes por presupuesto ---")
for b in budget_order:
    freq = {k: v for k, v in top_changes_budget.get(b, {}).items() if v > 0}
    low5 = sorted(freq.items(), key=lambda t: t[1])[:5]
    pretty = ", ".join([f"{col} ({cnt})" for col, cnt in low5]) if low5 else "N/A"
    lines.append(f"{b}: {pretty}")

lines.append("")
lines.append("--- Cambios más frecuentes por percentil ---")
for p in pct_order:
    freq = top_changes_pct.get(p, {})
    top5 = sorted(freq.items(), key=lambda t: t[1], reverse=True)[:5]
    pretty = ", ".join([f"{col} ({cnt})" for col, cnt in top5]) if top5 else "N/A"
    lines.append(f"{p}: {pretty}")

lines.append("")
lines.append("--- Cambios menos frecuentes por percentil ---")
for p in pct_order:
    freq = {k: v for k, v in top_changes_pct.get(p, {}).items() if v > 0}
    low5 = sorted(freq.items(), key=lambda t: t[1])[:5]
    pretty = ", ".join([f"{col} ({cnt})" for col, cnt in low5]) if low5 else "N/A"
    lines.append(f"{p}: {pretty}")

# Estadísticas de áreas/ambientes (global)
if area_records:
    area_df = pd.DataFrame(area_records)
    lines.append("")
    lines.append("--- Estadísticas de áreas/ambientes (media base | óptimo | delta) ---")
    for f in numeric_fields:
        b_col, o_col, d_col = f"base::{f}", f"opt::{f}", f"delta::{f}"
        b_mean = area_df[b_col].mean() if b_col in area_df else float("nan")
        o_mean = area_df[o_col].mean() if o_col in area_df else float("nan")
        d_mean = area_df[d_col].mean() if d_col in area_df else float("nan")
        lines.append(f"{f}: base={b_mean:,.2f} | opt={o_mean:,.2f} | Δ={d_mean:,.2f}")

    # Estadísticas por categoría/presupuesto/percentil
    lines.append("")
    lines.append("--- Áreas por categoría de barrio (media base | opt | delta) ---")
    for cat in ["baja", "media", "alta"]:
        sub = area_df.loc[area_df["neigh_cat"] == cat]
        if sub.empty:
            continue
        for f in numeric_fields:
            b_col, o_col, d_col = f"base::{f}", f"opt::{f}", f"delta::{f}"
            b_mean = sub[b_col].mean() if b_col in sub else float("nan")
            o_mean = sub[o_col].mean() if o_col in sub else float("nan")
            d_mean = sub[d_col].mean() if d_col in sub else float("nan")
            lines.append(f"{cat} | {f}: base={b_mean:,.2f} | opt={o_mean:,.2f} | Δ={d_mean:,.2f}")

    lines.append("")
    lines.append("--- Áreas por presupuesto (media base | opt | delta) ---")
    for b in sorted([x for x in area_df.get("budget", []) if pd.notna(x)], key=lambda x: float(x)):
        sub = area_df.loc[area_df["budget"] == b]
        for f in numeric_fields:
            b_col, o_col, d_col = f"base::{f}", f"opt::{f}", f"delta::{f}"
            b_mean = sub[b_col].mean() if b_col in sub else float("nan")
            o_mean = sub[o_col].mean() if o_col in sub else float("nan")
            d_mean = sub[d_col].mean() if d_col in sub else float("nan")
            lines.append(f"{int(float(b))} | {f}: base={b_mean:,.2f} | opt={o_mean:,.2f} | Δ={d_mean:,.2f}")

    lines.append("")
    lines.append("--- Áreas por percentil (media base | opt | delta) ---")
    for p in sorted([x for x in area_df.get("percentile", []) if pd.notna(x)]):
        sub = area_df.loc[area_df["percentile"] == p]
        for f in numeric_fields:
            b_col, o_col, d_col = f"base::{f}", f"opt::{f}", f"delta::{f}"
            b_mean = sub[b_col].mean() if b_col in sub else float("nan")
            o_mean = sub[o_col].mean() if o_col in sub else float("nan")
            d_mean = sub[d_col].mean() if d_col in sub else float("nan")
            lines.append(f"{p} | {f}: base={b_mean:,.2f} | opt={o_mean:,.2f} | Δ={d_mean:,.2f}")

    # Heatmaps de áreas (delta % vs base)
    save_area_heatmap(area_df, "neigh_cat", "Áreas Δ% por categoría", HEATMAP_DIR / "heatmap_areas_by_cat.png")
    save_area_heatmap(area_df, "budget", "Áreas Δ% por presupuesto", HEATMAP_DIR / "heatmap_areas_by_budget.png")
    save_area_heatmap(area_df, "percentile", "Áreas Δ% por percentil", HEATMAP_DIR / "heatmap_areas_by_percentil.png")
    save_area_bars(area_df, "neigh_cat", "cat")
    save_area_bars(area_df, "budget", "budget")
    save_area_bars(area_df, "percentile", "pct")

OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
print(f"Guardado resumen en {OUT_TXT}")
