#!/usr/bin/env python3
"""
ST_fitting/run_st_fitting.py – CLI entrypoint for regression-based
solar-thermal fitting from 5-minute data.

Usage
-----
    python ST_fitting/run_st_fitting.py \
        --csv data/FullDS_Findhorn_5min.csv \
        --yaml column_mapping_5min.yaml \
        --gti-csv data/hist_GTI_5min.csv \
        --plot
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data_loader import load_and_clean
from src.solar_thermal import compute_st_energy

from ST_fitting.config import STFitConfig
from ST_fitting.detector import detect_st_windows
from ST_fitting.evaluate import (
    plot_regression_diagnostics,
    plot_freeforward_windows,
    write_interval_csv,
    write_window_csv,
)
from ST_fitting.fit_st import fit_q_sol_regression

logger = logging.getLogger("ST_fitting")

_ST_DIR = Path(__file__).resolve().parent
_OUTPUT_DIR = _ST_DIR / "output"
_DIAG_DIR = _ST_DIR / "diagnostics"
_PLOT_DIR = _OUTPUT_DIR / "plots"

_DEFAULT_GTI_CSV = _REPO_ROOT / "data" / "hist_GTI_5min.csv"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit regression-based ST energy and flow-temperature models.",
    )
    p.add_argument("--csv", required=True, help="Path to the cleaned CSV.")
    p.add_argument("--yaml", required=True, help="Path to column_mapping.yaml.")
    p.add_argument("--gti-csv", type=str, default=None,
                   help="Path to GTI CSV (default: data/hist_GTI_5min.csv).")

    # Threshold overrides
    p.add_argument("--flow-dt", type=float, default=None)
    p.add_argument("--flow-min-l", type=float, default=None)
    p.add_argument("--power-min", type=float, default=None)
    p.add_argument("--ashp-off", type=float, default=None)
    p.add_argument("--imm-off", type=float, default=None)
    p.add_argument("--min-intervals", type=int, default=None)
    p.add_argument("--train-frac", type=float, default=None)

    p.add_argument("--plot", action="store_true",
                   help="Generate diagnostic plots.")
    p.add_argument("--n-plot", type=int, default=None,
                   help="Number of free-forward windows to plot (default 1).")
    return p.parse_args()


def _build_config(args: argparse.Namespace) -> STFitConfig:
    cfg = STFitConfig()
    if args.flow_dt is not None:
        cfg.st_flow_dt_min_c = args.flow_dt
    if args.flow_min_l is not None:
        cfg.st_flow_min_l = args.flow_min_l
    if args.power_min is not None:
        cfg.st_power_min_kw = args.power_min
    if args.ashp_off is not None:
        cfg.ashp_off_kwh = args.ashp_off
    if args.imm_off is not None:
        cfg.imm_off_kwh = args.imm_off
    if args.min_intervals is not None:
        cfg.min_st_intervals = args.min_intervals
    if args.train_frac is not None:
        cfg.train_frac = args.train_frac
    if args.n_plot is not None:
        cfg.n_plot_windows = args.n_plot
    return cfg


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-18s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()
    cfg = _build_config(args)
    gti_csv = Path(args.gti_csv) if args.gti_csv else _DEFAULT_GTI_CSV

    # -- 1. Load data --------------------------------------------------------
    logger.info("Loading data from %s ...", args.csv)
    df = load_and_clean(args.csv, args.yaml, sampling_minutes=cfg.sampling_minutes)

    # -- 2. Compute ST energy ------------------------------------------------
    df["st_kwh"] = compute_st_energy(df, dt_minutes=float(cfg.sampling_minutes))

    # -- 3. Detect ST-only windows -------------------------------------------
    logger.info("Detecting ST-only windows ...")
    windows, diag_df = detect_st_windows(df, cfg)

    _DIAG_DIR.mkdir(parents=True, exist_ok=True)
    write_interval_csv(diag_df, output_path=_DIAG_DIR / "st_intervals.csv")

    if not windows:
        logger.error("No ST-only windows found.")
        return 1

    # -- 4. Regression fitting -----------------------------------------------
    logger.info("Fitting Q_sol and T_flow regressions from %d windows ...", len(windows))
    result, reg_df = fit_q_sol_regression(
        df, windows, gti_csv, cfg, output_dir=_OUTPUT_DIR,
    )

    if "error" in result.get("identification", {}):
        logger.error("Fitting failed: %s", result["identification"]["error"])
        return 1

    # Save regression data CSV
    reg_csv = _DIAG_DIR / "st_regression_data.csv"
    reg_df.to_csv(reg_csv, index=False)
    logger.info("Regression data saved to %s", reg_csv)

    # Save window summary CSV
    write_window_csv(
        _make_window_df(windows),
        output_path=_DIAG_DIR / "st_windows.csv",
    )

    # -- 5. Diagnostic plots -------------------------------------------------
    if args.plot:
        _PLOT_DIR.mkdir(parents=True, exist_ok=True)
        plot_regression_diagnostics(reg_df, result, output_path=_PLOT_DIR / "regression_diagnostics.png")
        plot_freeforward_windows(df, windows, result, cfg, output_dir=_PLOT_DIR)
        logger.info("Plots saved to %s", _PLOT_DIR)

    logger.info("ST fitting pipeline complete.")
    return 0


def _make_window_df(windows):
    import pandas as pd
    records = []
    for w in windows:
        records.append({
            "window_id": w.window_id,
            "start": w.start,
            "end": w.end,
            "n_intervals": w.n_intervals,
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    sys.exit(main())
