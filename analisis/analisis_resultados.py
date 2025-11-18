"""Ejecutor masivo de optimizaciones para distintos presupuestos."""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from analisis.config_analisis import ConfigAnalisis
from optimization.remodel.benchmark_remodel import run_once
from optimization.remodel.config import PATHS
from optimization.remodel.io import load_base_df


def _parse_presupuestos(values: Iterable[str] | str | None) -> list[float]:
    if values is None:
        return list(ConfigAnalisis.PRESUPUESTOS)
    if isinstance(values, str):
        values = values.split(",")
    budgets: list[float] = []
    for raw in values:
        txt = str(raw).strip()
        if not txt:
            continue
        try:
            budgets.append(float(txt))
        except ValueError:
            raise ValueError(f"Presupuesto inválido: {raw!r}") from None
    if not budgets:
        raise ValueError("Debe especificar al menos un presupuesto.")
    return budgets


def _sample_pids(df: pd.DataFrame, n: int, seed: int) -> list[int]:
    if "PID" not in df.columns:
        raise ValueError("El CSV base no contiene columna 'PID'.")
    pids = [int(pid) for pid in df["PID"].dropna().unique().tolist()]
    if not pids:
        raise ValueError("No se encontraron PID válidos en la base.")
    random.seed(seed)
    random.shuffle(pids)
    return pids[: min(n, len(pids))]


def _agrupar_metricas(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grouped = (
        df.groupby("presupuesto")
        .agg(
            n_casas=("pid", "count"),
            pid_unicos=("pid", "nunique"),
            objective_mean=("objective_mip", "mean"),
            objective_max=("objective_mip", "max"),
            roi_usd_mean=("roi_usd", "mean"),
            roi_pct_mean=("roi_pct", "mean"),
            costo_medio=("budget_usado", "mean"),
            total_cost_mean=("total_cost", "mean"),
            precio_base_mean=("price_base", "mean"),
            precio_opt_mean=("price_opt", "mean"),
        )
        .reset_index()
    )
    grouped["mejor_pid_roi"] = None
    grouped["mejor_neighborhood"] = None
    grouped["roi_usd_top"] = None
    for idx, presupuesto in enumerate(grouped["presupuesto"].tolist()):
        subset = df[df["presupuesto"] == presupuesto]
        if subset.empty or subset["roi_usd"].isna().all():
            continue
        best_row = subset.loc[subset["roi_usd"].idxmax()]
        grouped.at[idx, "mejor_pid_roi"] = int(best_row["pid"])
        grouped.at[idx, "mejor_neighborhood"] = best_row.get("neighborhood")
        grouped.at[idx, "roi_usd_top"] = best_row.get("roi_usd")
    return grouped


def ejecutar_estudio(
    num_casas: int,
    presupuestos: list[float],
    base_csv: Path,
    seed: int,
    py_exec: str,
    time_limit: float | None,
) -> tuple[Path, Path]:
    base_df = load_base_df(base_csv)
    pids = _sample_pids(base_df, num_casas, seed)
    result_dir = Path("analisis") / "resultados"
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resultados_path = result_dir / f"{timestamp}_{ConfigAnalisis.ARCHIVO_RESULTADOS}"
    metricas_path = result_dir / f"{timestamp}_{ConfigAnalisis.ARCHIVO_METRICAS}"
    logdir = result_dir / f"logs_{timestamp}"
    logdir.mkdir(parents=True, exist_ok=True)

    print(f"Seleccionadas {len(pids)} casas únicas para el estudio.")
    print(f"Presupuestos a evaluar: {', '.join(f'${b:,.0f}' for b in presupuestos)}")
    resultados: list[dict] = []
    idx_global = 0
    for idx_pid, pid in enumerate(pids, start=1):
        print(f"\n🏠 Casa {idx_pid}/{len(pids)} | PID={pid}")
        for budget in presupuestos:
            idx_global += 1
            tier = f"budget_{int(round(budget))}"
            try:
                res = run_once(
                    pid=pid,
                    budget=budget,
                    py_exe=py_exec,
                    logdir=logdir,
                    tier=tier,
                    idx=idx_global,
                    basecsv=str(base_csv),
                    time_limit=time_limit,
                )
                error_msg = None if res.get("raw_ok") else "Salida incompleta (ver logs)"
                resultados.append(
                    {
                        "pid": pid,
                        "presupuesto": budget,
                        "neighborhood": res.get("neighborhood"),
                        "objective_mip": res.get("objective_mip"),
                        "roi_usd": res.get("roi"),
                        "roi_pct": res.get("roi_pct"),
                        "pct_net_improve": res.get("pct_net_improve"),
                        "price_base": res.get("price_base"),
                        "price_opt": res.get("price_opt"),
                        "delta_price": res.get("delta_price"),
                        "budget_usado": res.get("budget_used"),
                        "total_cost": res.get("total_cost"),
                        "slack": res.get("slack"),
                        "runtime_s": res.get("runtime_s"),
                        "mip_gap_pct": res.get("mip_gap_pct"),
                        "bins_active": res.get("bins_active"),
                        "error": error_msg,
                    }
                )
                estado = "OK" if error_msg is None else "WARN"
                objetivo = res.get("objective_mip")
                print(
                    f"  - Presupuesto ${budget:,.0f}: obj={objetivo:.1f} ROI$={res.get('roi')} [{estado}]"
                    if objetivo is not None
                    else f"  - Presupuesto ${budget:,.0f}: ejecución sin métricas [{estado}]"
                )
            except Exception as exc:
                resultados.append(
                    {
                        "pid": pid,
                        "presupuesto": budget,
                        "neighborhood": None,
                        "objective_mip": None,
                        "roi_usd": None,
                        "roi_pct": None,
                        "pct_net_improve": None,
                        "price_base": None,
                        "price_opt": None,
                        "delta_price": None,
                        "budget_usado": None,
                        "total_cost": None,
                        "slack": None,
                        "runtime_s": None,
                        "mip_gap_pct": None,
                        "bins_active": None,
                        "error": str(exc),
                    }
                )
                print(f"  - Presupuesto ${budget:,.0f}: ERROR {exc}")

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(resultados_path, index=False)
    print(f"\n📄 Resultados detallados guardados en {resultados_path}")

    df_ok = df_resultados[df_resultados["error"].isna()]
    metricas = _agrupar_metricas(df_ok)
    if not metricas.empty:
        metricas.to_csv(metricas_path, index=False)
        print(f"📊 Métricas agregadas guardadas en {metricas_path}")
    else:
        print("⚠ No se pudieron calcular métricas agregadas (sin ejecuciones exitosas).")

    return resultados_path, metricas_path


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Corre múltiples optimizaciones por presupuesto.")
    ap.add_argument(
        "--num-casas",
        type=int,
        default=ConfigAnalisis.NUM_CASAS,
        help="Cantidad de casas distintas a evaluar.",
    )
    ap.add_argument(
        "--presupuestos",
        type=str,
        default=",".join(str(p) for p in ConfigAnalisis.PRESUPUESTOS),
        help="Lista de presupuestos separada por comas.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=ConfigAnalisis.RANDOM_SEED,
        help="Semilla para la selección de casas.",
    )
    ap.add_argument(
        "--basecsv",
        type=str,
        default=str(PATHS.base_csv),
        help="Ruta al CSV base con las casas.",
    )
    ap.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Intérprete de Python a usar para llamar a run_opt.",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=ConfigAnalisis.TIMEOUT_EJECUCION,
        help="Límite de tiempo (segundos) por corrida del solver.",
    )
    return ap


def main():
    parser = _build_parser()
    args = parser.parse_args()
    presupuestos = _parse_presupuestos(args.presupuestos)
    ejecutar_estudio(
        num_casas=args.num_casas,
        presupuestos=presupuestos,
        base_csv=Path(args.basecsv),
        seed=args.seed,
        py_exec=args.python,
        time_limit=args.timeout,
    )


if __name__ == "__main__":
    main()
