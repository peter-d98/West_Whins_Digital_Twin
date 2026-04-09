#!/usr/bin/env python3
"""
run_stage1.py – Stage-1 Digital Twin Evaluation Runner
======================================================

Loads all frozen priors from the three fitting pipelines (UA_fitting,
ASHP_fitting, Global_fitting), runs a forward simulation over the full
30-min dataset, and reports per-node RMSE/MAE for train and validation
splits.

Usage
-----
    python run_stage1.py                          # defaults
    python run_stage1.py --csv path/to/data.csv   # custom paths
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from src import data_loader
from src.ashp_model import ASHPParams, predict_cop, sink_proxy
from src.solar_thermal import compute_st_energy
from src.tank_model import TankParams, simulate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent

# Default prior paths
_UA_FIT_PATH     = ROOT / "UA_fitting" / "output" / "ua_fit.json"
_ASHP_FIT_PATH   = ROOT / "ASHP_fitting" / "output" / "ashp_fit.json"
_GLOBAL_FIT_PATH = ROOT / "Global_fitting" / "output" / "global_fit.json"


def _load_priors(
    ua_path: Path,
    ashp_path: Path,
    global_path: Path,
) -> tuple[TankParams, ASHPParams]:
    """Load frozen priors from all three fitting pipelines.

    Raises FileNotFoundError with a clear message listing which pipeline
    to run if any prior JSON is missing.
    """
    missing = []
    if not ua_path.exists():
        missing.append(
            f"  - {ua_path}\n"
            f"    Run: python UA_fitting/run_ua_fitting.py --csv <csv> --yaml <yaml>"
        )
    if not ashp_path.exists():
        missing.append(
            f"  - {ashp_path}\n"
            f"    Run: python ASHP_fitting/run_ashp_fitting.py --csv <csv> --yaml <yaml>"
        )
    if not global_path.exists():
        missing.append(
            f"  - {global_path}\n"
            f"    Run: python Global_fitting/run_global_fitting.py --csv <csv> --yaml <yaml>"
        )
    if missing:
        raise FileNotFoundError(
            "Missing prior JSON file(s):\n" + "\n".join(missing)
        )

    # Load global fit (contains tank params)
    with open(global_path, "r", encoding="utf-8") as f:
        gdata = json.load(f)

    tank_params = TankParams()
    tank_params.UA_loss = np.array(gdata["UA_loss"], dtype=float)
    tank_params.UA_adj  = np.array(gdata["UA_adj"], dtype=float)
    tank_params.f_st    = np.array(gdata["f_st"], dtype=float)
    tank_params.f_ashp  = np.array(gdata["f_ashp"], dtype=float)
    tank_params.f_imm   = np.array(gdata["f_imm"], dtype=float)

    # Load ASHP params
    with open(ashp_path, "r", encoding="utf-8") as f:
        adata = json.load(f)
    ashp_params = ASHPParams(
        c=np.array(adata["ashp"]["c"], dtype=float) if "c" in adata["ashp"] else None,
        a=np.array(adata["ashp"]["a"], dtype=float) if "a" in adata["ashp"] else ASHPParams().a,
        b=np.array(adata["ashp"]["b"], dtype=float) if "b" in adata["ashp"] else ASHPParams().b,
    )

    return tank_params, ashp_params


def _prepare_inputs(df, ashp_params: ASHPParams, dt_h: float | None = None,
                    sampling_minutes: int = 5) -> dict:
    """Build arrays needed for simulation."""
    if dt_h is None:
        dt_h = sampling_minutes / 60.0
    T_meas = df[["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]].values

    if "st_kwh" in df.columns:
        Q_st = df["st_kwh"].fillna(0).values
    else:
        Q_st = compute_st_energy(df, dt_minutes=dt_h * 60).values

    T_sink = sink_proxy(df["tank_mid_c"].values, df["tank_top_c"].values)
    T_out = df["t_out_c"].fillna(df["t_out_c"].median()).values
    cop = predict_cop(T_out, T_sink, ashp_params)
    P_meas = df["ashp_inst_kwh"].fillna(0).values
    ashp_on = P_meas > 0.016
    Q_ashp = np.where(ashp_on, P_meas * cop, 0.0)

    Q_imm = df["imm_tot_inst_kwh"].fillna(0).values
    T_amb = df["t_amb_c"].fillna(df["t_amb_c"].median()).values

    return dict(T_meas=T_meas, Q_st=Q_st, Q_ashp=Q_ashp, Q_imm=Q_imm, T_amb=T_amb)


def _evaluate_split(
    df, tank_params: TankParams, ashp_params: ASHPParams, label: str,
    sampling_minutes: int = 5,
) -> dict:
    """Simulate and compute per-node RMSE and MAE."""
    inputs = _prepare_inputs(df, ashp_params, sampling_minutes=sampling_minutes)
    T_meas = inputs["T_meas"]
    dt_s = sampling_minutes * 60.0

    T_hist = simulate(
        T_meas[0],
        inputs["Q_st"],
        inputs["Q_ashp"],
        inputs["Q_imm"],
        inputs["T_amb"],
        tank_params,
        dt_s,
    )

    # T_hist has shape (N+1, 4), compare T_hist[1:] vs T_meas[1:]
    # (skip first step which is the initial condition)
    N = len(T_meas) - 1
    T_sim = T_hist[1:N + 1]
    T_actual = T_meas[1:]

    node_names = ["T_bottom", "T_mid", "T_mid_hi", "T_top"]
    node_rmse = {}
    node_mae = {}
    for i, name in enumerate(node_names):
        mask = np.isfinite(T_actual[:, i]) & np.isfinite(T_sim[:, i])
        if mask.sum() == 0:
            node_rmse[name] = float("nan")
            node_mae[name] = float("nan")
            continue
        err = T_sim[mask, i] - T_actual[mask, i]
        node_rmse[name] = float(np.sqrt(np.mean(err ** 2)))
        node_mae[name] = float(np.mean(np.abs(err)))

    logger.info("=== %s evaluation ===", label.upper())
    for name in node_names:
        logger.info("  RMSE %s: %.2f °C  |  MAE: %.2f °C",
                     name, node_rmse[name], node_mae[name])

    return {
        "label": label,
        "node_rmse": node_rmse,
        "node_mae": node_mae,
        "n_samples": len(df),
    }


def main(
    csv_path: Path | None = None,
    yaml_path: Path | None = None,
    output_dir: Path | None = None,
    train_frac: float = 0.7,
    ua_path: Path | None = None,
    ashp_path: Path | None = None,
    global_path: Path | None = None,
) -> dict:
    csv_path    = csv_path    or ROOT / "data" / "FullDS_Findhorn_5min.csv"
    yaml_path   = yaml_path   or ROOT / "column_mapping_5min.yaml"
    output_dir  = output_dir  or ROOT / "output"
    ua_path     = ua_path     or _UA_FIT_PATH
    ashp_path   = ashp_path   or _ASHP_FIT_PATH
    global_path = global_path or _GLOBAL_FIT_PATH
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load frozen priors -----------------------------------------------
    try:
        tank_params, ashp_params = _load_priors(ua_path, ashp_path, global_path)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    logger.info("Loaded frozen priors from all three fitting pipelines.")

    # ---- Load & clean data ------------------------------------------------
    logger.info("Loading data from %s", csv_path)
    cfg = data_loader.load_column_mapping(yaml_path)
    sampling_minutes = cfg.get("assumptions", {}).get("sampling_minutes", 5)
    df = data_loader.load_and_clean(csv_path, yaml_path,
                                    sampling_minutes=sampling_minutes)

    tank_cols = ["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]
    df = df.dropna(subset=tank_cols, how="all")
    logger.info("After dropping all-NaN tank rows: %d rows", len(df))

    # Compute ST energy
    df["st_kwh"] = compute_st_energy(df, dt_minutes=float(sampling_minutes))

    # ---- Node-ordering diagnostic -----------------------------------------
    ordering = data_loader.node_ordering_check(df)
    logger.info("Node ordering satisfied: %.1f %%", ordering.mean() * 100)

    # ---- Train/val split --------------------------------------------------
    split_idx = int(len(df) * train_frac)
    df_train = df.iloc[:split_idx]
    df_val = df.iloc[split_idx:]
    logger.info("Train: %d rows, Val: %d rows", len(df_train), len(df_val))

    # ---- Evaluate ---------------------------------------------------------
    summary_train = _evaluate_split(df_train, tank_params, ashp_params, "train",
                                    sampling_minutes=sampling_minutes)
    summary_val = _evaluate_split(df_val, tank_params, ashp_params, "validation",
                                  sampling_minutes=sampling_minutes)

    summary = {"train": summary_train, "val": summary_val}
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    logger.info("Summary saved to %s", summary_file)

    return summary


def _json_default(obj):
    """JSON serialiser fallback for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage-1 DHW Digital Twin — Evaluation Runner"
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--yaml", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--train-frac", type=float, default=0.7)
    args = parser.parse_args()
    main(args.csv, args.yaml, args.output, args.train_frac)
