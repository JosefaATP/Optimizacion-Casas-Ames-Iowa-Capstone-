from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(BASE_DIR / name)


def plot_budget_prices(df: pd.DataFrame) -> None:
    budgets = df["budget"] / 1000  # kUSD for axis readability
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(budgets, df["avg_price_xgb"], marker="o", label="Precio XGB")
    ax.plot(budgets, df["avg_price_reg"], marker="s", label="Precio Regresión")
    ax2 = ax.twinx()
    ax2.plot(budgets, df["avg_delta_abs"], marker="^", linestyle="--", color="black", label="Brecha XGB-Reg")
    ax.set_xlabel("Presupuesto (kUSD)")
    ax.set_ylabel("Precio promedio (USD)")
    ax2.set_ylabel("Brecha (USD)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "budget_prices.png", dpi=300)
    plt.close(fig)


def plot_neighborhood_gap(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(df["neighborhood"], df["avg_delta_abs"], color="#377eb8")
    ax.set_ylabel("Gap promedio (USD)")
    ax.set_xlabel("Barrio")
    ax.set_xticks(range(len(df["neighborhood"])))
    ax.set_xticklabels(df["neighborhood"], rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gap_by_neighborhood.png", dpi=300)
    plt.close(fig)


def plot_roi_hist(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["bucket"], df["count"], color="#4daf4a")
    ax.set_xlabel("ROI (intervalos)")
    ax.set_ylabel("Frecuencia")
    ax.set_xticks(range(len(df["bucket"])))
    ax.set_xticklabels(df["bucket"], rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roi_hist.png", dpi=300)
    plt.close(fig)


def plot_negative_by_budget(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(df["budget"], df["neg_runs"], color="#e41a1c")
    ax.set_xlabel("Presupuesto (USD)")
    ax.set_ylabel("Casos ROI < 0")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    for x, y in zip(df["budget"], df["neg_runs"]):
        ax.text(x, y + 0.5, f"{int(y)}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roi_negative_by_budget.png", dpi=300)
    plt.close(fig)


def plot_price_by_lot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.plot(df["lot_area"], df["avg_price_xgb"], marker="o", color="#984ea3")
    ax.set_xlabel("Lot Area (ft²)")
    ax.set_ylabel("Precio XGB promedio (USD)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "price_by_lot.png", dpi=300)
    plt.close(fig)


def plot_features_vs_baseline(df: pd.DataFrame) -> None:
    metrics = [
        ("Gr Liv Area", "bench_gr_liv_area", "base_gr_liv_area"),
        ("Total Bsmt", "bench_bsmt", "base_bsmt"),
        ("Garage Area", "bench_garage_area", "base_garage_area"),
        ("Dormitorios", "bench_beds", "base_beds"),
        ("Baños completos", "bench_fullbath", "base_fullbath"),
        ("Overall Qual", "bench_overall", "base_overall"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax, (title, bench_col, base_col) in zip(axes.flatten(), metrics):
        x = range(len(df))
        width = 0.35
        ax.bar([i - width / 2 for i in x], df[bench_col], width=width, label="Benchmark", color="#377eb8")
        ax.bar([i + width / 2 for i in x], df[base_col], width=width, label="Base", color="#ff7f00")
        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.set_xticklabels(df["neighborhood"], rotation=45, ha="right", fontsize=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "features_vs_baseline.png", dpi=300)
    plt.close(fig)


def plot_delta_pct_histogram(runs_df: pd.DataFrame) -> None:
    df = runs_df[runs_df["status"] == 2].copy()
    df["delta_pct"] = df["delta_reg_pct"]
    budgets = sorted(df["budget"].unique())
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = 15
    colors = ["#377eb8", "#ff7f00", "#4daf4a", "#984ea3", "#a65628"]
    for budget, color in zip(budgets, colors):
        subset = df[df["budget"] == budget]["delta_pct"]
        if subset.empty:
            continue
        ax.hist(subset, bins=bins, alpha=0.4, label=f"{int(budget):,}$", color=color)
    ax.set_xlabel("Brecha porcentual XGB - Regresión (%)")
    ax.set_ylabel("Frecuencia")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Presupuesto")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "delta_pct_by_budget.png", dpi=300)
    plt.close(fig)


def plot_mip_scenario_summary(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return
    df = summary_df.sort_values("max_roi_pct", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["scenario"], df["max_roi_pct"], color="#5a9bd5")
    ax.set_ylabel("ROI máximo (%)")
    ax.set_xlabel("Escenario MIP")
    ax.set_xticks(range(len(df["scenario"])))
    ax.set_xticklabels(df["scenario"], rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mip_scenario_roi_summary.png", dpi=300)
    plt.close(fig)


def plot_mip_roi_vs_budget(detail_df: pd.DataFrame) -> None:
    if detail_df.empty:
        return
    keep = [
        "starter_1fam",
        "moveup_1fam",
        "premium_1fam",
        "compact_duplex_highqual",
        "premium_duplex",
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for scen in keep:
        sdf = detail_df[detail_df["scenario"] == scen].sort_values("budget")
        if sdf.empty:
            continue
        label = scen.replace("_", " ").title()
        ax.plot(sdf["budget"], sdf["roi_pct"], marker="o", label=label)
    ax.set_xlabel("Presupuesto (USD)")
    ax.set_ylabel("ROI (%)")
    ax.grid(True, axis="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mip_roi_vs_budget.png", dpi=300)
    plt.close(fig)


def main() -> None:
    budget = load_csv("price_by_budget.csv")
    neighborhood = load_csv("delta_by_neighborhood_top.csv")
    roi_hist_df = load_csv("roi_hist.csv")
    lot_df = load_csv("price_by_lot.csv")
    features_df = load_csv("features_vs_baseline_top.csv")
    runs_df = pd.read_csv(BASE_DIR.parent / "benchmark_runs.csv")
    mip_summary_path = BASE_DIR / "mip_scenarios_summary.csv"
    mip_detail_path = BASE_DIR / "mip_scenarios_roi_by_budget.csv"

    plot_budget_prices(budget)
    plot_neighborhood_gap(neighborhood)
    plot_roi_hist(roi_hist_df)
    plot_negative_by_budget(budget)
    plot_price_by_lot(lot_df)
    plot_features_vs_baseline(features_df.iloc[:6])
    plot_delta_pct_histogram(runs_df)
    if mip_summary_path.exists():
        mip_summary = pd.read_csv(mip_summary_path)
        plot_mip_scenario_summary(mip_summary)
    if mip_detail_path.exists():
        mip_detail = pd.read_csv(mip_detail_path)
        plot_mip_roi_vs_budget(mip_detail)


if __name__ == "__main__":
    main()
