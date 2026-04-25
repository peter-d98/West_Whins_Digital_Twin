#!/usr/bin/env python3
"""
ST_fitting/run_st_fitting.py – CLI entrypoint for solar-thermal window
detection and fitting from 30-minute plant data.

Usage
-----
    # Windows-only (detect + back-calculate, no regression):
    python ST_fitting/run_st_fitting.py \\
        --csv data/FullDS_Findhorn.csv \\
        --yaml column_mapping.yaml \\
        --windows-only

    # Full pipeline (detection + regression fitting):
    python ST_fitting/run_st_fitting.py \\
        --csv data/FullDS_Findhorn.csv \\
        --yaml column_mapping.yaml

Output is written to ST_fitting/diagnostics/ and ST_fitting/output/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data_loader import load_and_clean
from src.tank_model import NODE_CAP

from ST_fitting.config import STFitConfig
from ST_fitting.detector import detect_st_windows
from ST_fitting.evaluate import write_interval_csv, write_window_csv

logger = logging.getLogger("ST_fitting")

# Output directories (relative to ST_fitting/)
_ST_DIR = Path(__file__).resolve().parent
_OUTPUT_DIR = _ST_DIR / "output"
_DIAG_DIR = _ST_DIR / "diagnostics"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect ST-only charging windows and optionally fit "
                    "regression models from 30-minute plant data.",
    )
    p.add_argument("--csv", required=True, help="Path to the plant CSV.")
    p.add_argument("--yaml", required=True, help="Path to column_mapping.yaml.")

    # Threshold overrides
    p.add_argument("--power-start", type=float, default=None,
                   help="Min ST power to open a window [kW] (default 0.5).")
    p.add_argument("--rise-start", type=float, default=None,
                   help="Min bottom-node rise to open a window [°C] (default 2.0).")
    p.add_argument("--ashp-off", type=float, default=None,
                   help="ASHP-off threshold [kWh/interval] (default 0.016).")
    p.add_argument("--imm-off", type=float, default=None,
                   help="Immersion-off threshold [kWh/interval] (default 0.001).")
    p.add_argument("--flat-close", type=int, default=None,
                   help="Consecutive non-rising intervals to close (default 2).")
    p.add_argument("--min-intervals", type=int, default=None,
                   help="Minimum intervals per window (default 2).")
    p.add_argument("--train-frac", type=float, default=None,
                   help="Training fraction (default 0.7).")
    p.add_argument("--sampling-minutes", type=int, default=None,
                   help="Sampling cadence [minutes] (default 30).")

    # Modes
    p.add_argument("--windows-only", action="store_true",
                   help="Detect windows and output CSV only (skip fitting).")

    return p.parse_args()


def _build_config(args: argparse.Namespace) -> STFitConfig:
    cfg = STFitConfig()
    if args.power_start is not None:
        cfg.st_power_start_kw = args.power_start
    if args.rise_start is not None:
        cfg.bottom_rise_start_c = args.rise_start
    if args.ashp_off is not None:
        cfg.ashp_off_kwh = args.ashp_off
    if args.imm_off is not None:
        cfg.imm_off_kwh = args.imm_off
    if args.flat_close is not None:
        cfg.bottom_flat_close = args.flat_close
    if args.min_intervals is not None:
        cfg.min_st_intervals = args.min_intervals
    if args.train_frac is not None:
        cfg.train_frac = args.train_frac
    if args.sampling_minutes is not None:
        cfg.sampling_minutes = args.sampling_minutes
    return cfg


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-18s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()
    cfg = _build_config(args)

    # -- 1. Load and clean data ----------------------------------------------
    logger.info("Loading %s with mapping %s (cadence=%d min) ...",
                args.csv, args.yaml, cfg.sampling_minutes)
    df = load_and_clean(args.csv, args.yaml, sampling_minutes=cfg.sampling_minutes)

    # -- 2. Detect ST-only windows -------------------------------------------
    logger.info("Detecting ST-only windows ...")
    windows, diag_df = detect_st_windows(df, cfg)

    # Save interval diagnostics
    _DIAG_DIR.mkdir(parents=True, exist_ok=True)
    write_interval_csv(diag_df, output_path=_DIAG_DIR / "st_intervals.csv")

    if not windows:
        logger.error("No ST-only windows found.  Check thresholds and data.")
        return 1

    # -- 3. Build window summary with Q metrics ------------------------------
    dt_h = cfg.sampling_minutes / 60.0
    n_train = int(len(df) * cfg.train_frac)
    df_train = df.iloc[:n_train]

    records = []
    for w in windows:
        # Q from power meter integration
        q_power = float(
            (df_train[cfg.st_power_col].fillna(0.0).iloc[w.indices] * dt_h).sum()
        )
        # Q back-calculated from all 4 node temperatures
        T_start = df_train[cfg.node_cols].iloc[w.indices[0]].values
        T_end = df_train[cfg.node_cols].iloc[w.indices[-1]].values
        q_backcalc = float(np.sum(NODE_CAP * (T_end - T_start)) / 3600.0)

        records.append({
            "window_id": w.window_id,
            "start": w.start,
            "end": w.end,
            "n_intervals": w.n_intervals,
            "Q_sol_power_kwh": round(q_power, 6),
            "Q_sol_backcalc_kwh": round(q_backcalc, 6),
        })

    window_df = pd.DataFrame(records)
    window_csv = _DIAG_DIR / "st_windows.csv"
    write_window_csv(window_df, output_path=window_csv)

    logger.info(
        "ST windows: %d  |  Q_power: %.3f–%.3f kWh  |  "
        "Q_backcalc: %.3f–%.3f kWh",
        len(windows),
        window_df["Q_sol_power_kwh"].min(),
        window_df["Q_sol_power_kwh"].max(),
        window_df["Q_sol_backcalc_kwh"].min(),
        window_df["Q_sol_backcalc_kwh"].max(),
    )

    if args.windows_only:
        logger.info("Windows-only mode: done.")
        return 0

    # -- 4. (Future) Regression fitting goes here ----------------------------
    logger.info("Regression fitting not yet implemented in 30-min pipeline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
