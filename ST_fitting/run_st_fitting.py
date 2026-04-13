#!/usr/bin/env python3
"""
ST_fitting/run_st_fitting.py – CLI entrypoint for solar-thermal regression
fitting (Q_sol energy map + T_flow bilinear map) from 5-minute data.

Usage
-----
    # Basic run (detection + Q_sol/T_flow regression, saves JSON and diagnostics):
    python ST_fitting/run_st_fitting.py \\
        --csv data/FullDS_Findhorn_5min.csv \\
        --yaml column_mapping_5min.yaml

    # With explicit GTI CSV and diagnostic plots:
    python ST_fitting/run_st_fitting.py \\
        --csv data/FullDS_Findhorn_5min.csv \\
        --yaml column_mapping_5min.yaml \\
        --gti-csv data/hist_GTI_5min.csv \\
        --plot

    # Override thresholds:
    python ST_fitting/run_st_fitting.py \\
        --csv data/FullDS_Findhorn_5min.csv \\
        --yaml column_mapping_5min.yaml \\
        --flow-dt 4.0 --flow-min-l 0.04 --power-min 0.1

All output is written under ST_fitting/output/ and ST_fitting/diagnostics/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Ensure the repository root is on sys.path so that both ``src.*`` and
# ``ST_fitting.*`` are importable when running this script directly.
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

# Output directories (relative to ST_fitting/)
_ST_DIR = Path(__file__).resolve().parent
_OUTPUT_DIR = _ST_DIR / "output"
_DIAG_DIR = _ST_DIR / "diagnostics"
_PLOT_DIR = _OUTPUT_DIR / "plots"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit Q_sol and T_flow regression maps from clean "
                    "ST-only windows in 5-minute data.",
    )
    p.add_argument("--csv", required=True, help="Path to the cleaned CSV.")
    p.add_argument("--yaml", required=True, help="Path to column_mapping.yaml.")
    p.add_argument(
        "--gti-csv", type=str, default=None,
        help="Path to hist_GTI_5min.csv for Q_sol regression "
             "(default: data/hist_GTI_5min.csv relative to repo root).",
    )
    p.add_argument(
        "--q-source",
        choices=["power", "derived"],
        default="power",
        help="Q target source for fitting: power meter or derived flow×deltaT (default: power).",
    )
    p.add_argument(
        "--no-q-benchmark",
        action="store_true",
        help="Disable side-by-side benchmark metrics between power and derived Q targets.",
    )

    # Threshold overrides
    p.add_argument("--flow-dt", type=float, default=None,
                   help="G1: min ST flow T minus tank bottom T [°C] (default 4.0).")
    p.add_argument("--flow-min-l", type=float, default=None,
                   help="G2: min ST flow volume per interval [L] (default 0.04).")
    p.add_argument("--power-min", type=float, default=None,
                   help="G3: min ST power [kW] (default 0.1).")
    p.add_argument("--ashp-off", type=float, default=None,
                   help="G5: ASHP-off threshold [kWh/interval] (default 0.016).")
    p.add_argument("--imm-off", type=float, default=None,
                   help="G6: immersion-off threshold [kWh/interval] (default 0.001).")
    p.add_argument("--min-intervals", type=int, default=None,
                   help="Minimum ST-only intervals per window (default 4).")
    p.add_argument("--train-frac", type=float, default=None,
                   help="Training fraction (default 0.7).")

    # Modes
    p.add_argument("--plot", action="store_true",
                   help="Generate regression and free-forward diagnostic plots.")
    p.add_argument("--n-plot", type=int, default=None,
                   help="Number of windows to plot (default 1).")

    return p.parse_args()


def _build_config(args: argparse.Namespace) -> STFitConfig:
    """Build an STFitConfig, applying any CLI overrides."""
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
    """Run the full ST fitting pipeline.  Returns 0 on success, 1 on failure."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-18s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()
    cfg = _build_config(args)

    # -- 1. Resolve GTI CSV path ---------------------------------------------
    gti_csv = Path(args.gti_csv) if args.gti_csv else (_REPO_ROOT / "data" / "hist_GTI_5min.csv")
    if not gti_csv.exists():
        logger.error("GTI CSV not found: %s", gti_csv)
        return 1

    # -- 2. Load and clean data ----------------------------------------------
    logger.info("Loading data from %s with mapping %s ...", args.csv, args.yaml)
    df = load_and_clean(args.csv, args.yaml, sampling_minutes=cfg.sampling_minutes)

    # -- 3. Compute solar-thermal energy -------------------------------------
    logger.info("Computing solar-thermal interval energy ...")
    df["st_kwh"] = compute_st_energy(df, dt_minutes=float(cfg.sampling_minutes))

    # -- 4. Detect ST-only windows -------------------------------------------
    logger.info("Detecting ST-only windows ...")
    windows, diag_df = detect_st_windows(df, cfg)

    # Save interval diagnostics CSV (always)
    _DIAG_DIR.mkdir(parents=True, exist_ok=True)
    diag_path = _DIAG_DIR / "st_intervals.csv"
    write_interval_csv(diag_df, output_path=diag_path)

    if not windows:
        logger.error("No ST-only windows found.  Check thresholds and data.")
        return 1

    # -- 5. Fit Q_sol and T_flow regressions ---------------------------------
    logger.info("Fitting Q_sol and T_flow from %d windows ...", len(windows))
    result, reg_df = fit_q_sol_regression(
        df,
        windows,
        gti_csv,
        cfg,
        q_source=args.q_source,
        benchmark_sources=not args.no_q_benchmark,
        output_dir=_OUTPUT_DIR,
    )

    if "error" in result.get("identification", {}):
        logger.error("ST fitting failed: %s", result["identification"]["error"])
        return 1

    logger.info(
        "Q_sol regression (%s source): q0=%.6f kWh  q1=%.6f kWh/(W/m²)  R²=%.4f  n=%d",
        result["identification"].get("q_source", args.q_source),
        result["Q_sol"]["q0_kwh"],
        result["Q_sol"]["q1_kwh_per_wm2"],
        result["Q_sol"]["r2"],
        result["Q_sol"]["n_points"],
    )
    q_bench = result.get("identification", {}).get("q_source_benchmark")
    if q_bench:
        logger.info(
            "Q benchmark: power R²=%.4f, derived R²=%.4f",
            q_bench["power"]["r2"],
            q_bench["derived"]["r2"],
        )
    logger.info(
        "T_flow regression: b0=%.4f°C  b1=%.4f  b2=%.6f °C/(W/m²)  R²=%.4f  n=%d",
        result["T_flow"]["b0_c"],
        result["T_flow"]["b1_per_c"],
        result["T_flow"]["b2_c_per_wm2"],
        result["T_flow"]["r2"],
        result["T_flow"]["n_points"],
    )

    # -- 6. Save regression diagnostics CSV ----------------------------------
    reg_csv_path = _DIAG_DIR / "st_regression_data.csv"
    reg_df.to_csv(reg_csv_path, index=True)
    logger.info("Saved regression data to %s", reg_csv_path)

    # -- 7. Save window-level diagnostics CSV --------------------------------
    # Build a simple window summary from windows list
    import pandas as pd
    window_records = []
    dt_h = cfg.sampling_minutes / 60.0
    for w in windows:
        n_train = int(len(df) * cfg.train_frac)
        df_train = df.iloc[:n_train]
        q_st_power_kwh = float(
            (df_train[cfg.st_power_col].fillna(0.0).iloc[w.indices] * dt_h).sum()
        )
        flow_l = df_train[cfg.st_flow_col].iloc[w.indices].fillna(0.0)
        t_flow = df_train[cfg.st_flow_temp_col].iloc[w.indices].fillna(0.0)
        t_ret = df_train[cfg.st_return_temp_col].iloc[w.indices].fillna(np.nan)
        q_st_derived_kwh = float(((flow_l * 4.186 * (t_flow - t_ret).clip(lower=0.0)) / 3600.0).sum())
        window_records.append({
            "window_id": w.window_id,
            "start": w.start,
            "end": w.end,
            "n_intervals": w.n_intervals,
            "Q_st_power_kwh": round(q_st_power_kwh, 6),
            "Q_st_derived_kwh": round(q_st_derived_kwh, 6),
            "Q_st_meas_kwh": round(q_st_power_kwh if args.q_source == "power" else q_st_derived_kwh, 6),
            "Q_source_used": args.q_source,
        })
    window_df = pd.DataFrame(window_records)
    window_csv_path = _DIAG_DIR / "st_windows.csv"
    write_window_csv(window_df, output_path=window_csv_path)

    # -- 8. Optional: diagnostic plots ---------------------------------------
    if args.plot:
        _PLOT_DIR.mkdir(parents=True, exist_ok=True)

        # Regression diagnostics
        plot_regression_diagnostics(reg_df, result, output_path=_PLOT_DIR / "regression_diagnostics.png")

        # Free-forward validation on longest windows
        sorted_windows = sorted(windows, key=lambda w: w.n_intervals, reverse=True)
        n_plot = min(cfg.n_plot_windows, len(sorted_windows))
        plot_freeforward_windows(
            df, sorted_windows[:n_plot], result, cfg,
            output_dir=_PLOT_DIR,
        )

        logger.info("Saved plots to %s", _PLOT_DIR)

    logger.info("ST fitting pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
