#!/usr/bin/env python3
"""
UA_fitting/run_ua_fitting.py – CLI entrypoint for the UA_loss fitting tool.

Usage
-----
    # Basic run (detection + fitting, saves JSON and diagnostics CSV):
    python UA_fitting/run_ua_fitting.py \
        --csv data/FullDS_Findhorn.csv \
        --yaml column_mapping.yaml

    # With diagnostic plots and QC CSV:
    python UA_fitting/run_ua_fitting.py \
        --csv data/FullDS_Findhorn.csv \
        --yaml column_mapping.yaml \
        --plot --qc-csv

    # Override thresholds:
    python UA_fitting/run_ua_fitting.py \
        --csv data/FullDS_Findhorn.csv \
        --yaml column_mapping.yaml \
        --ashp-off 0.05 --min-idle 3 --ridge 0.0

All output is written under UA_fitting/output/ and UA_fitting/diagnostics/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the repository root is on sys.path so that both ``src.*`` and
# ``UA_fitting.*`` are importable when running this script directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data_loader import load_and_clean
from src.solar_thermal import compute_st_energy

from UA_fitting.config import UAConfig
from UA_fitting.detector import detect_idle_windows
from UA_fitting.evaluate import plot_qc, write_qc_csv
from UA_fitting.fit_ua import fit_ua

logger = logging.getLogger("UA_fitting")

# Output directories (relative to UA_fitting/)
_UA_DIR = Path(__file__).resolve().parent
_OUTPUT_DIR = _UA_DIR / "output"
_DIAG_DIR = _UA_DIR / "diagnostics"
_PLOT_DIR = _OUTPUT_DIR / "plots"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit per-node UA_loss from idle tank periods.",
    )
    p.add_argument("--csv", required=True, help="Path to the cleaned CSV.")
    p.add_argument("--yaml", required=True, help="Path to column_mapping.yaml.")

    # Threshold overrides
    p.add_argument("--ashp-off", type=float, default=None,
                   help="ASHP-off threshold [kWh] (default 0.10).")
    p.add_argument("--st-off", type=float, default=None,
                   help="Solar-thermal-off threshold [kWh] (default 0.05).")
    p.add_argument("--imm-off", type=float, default=None,
                   help="Immersion-off threshold [kWh] (default 0.01).")
    p.add_argument("--min-idle", type=int, default=None,
                   help="Minimum idle intervals (default 2).")
    p.add_argument("--draw-delta", type=float, default=None,
                   help="Draw delta threshold [°C] (default -2.0).")
    p.add_argument("--min-windows", type=int, default=None,
                   help="Minimum number of idle windows (default 20).")
    p.add_argument("--ridge", type=float, default=None,
                   help="Ridge alpha (default 1e-4, 0 to disable).")
    p.add_argument("--train-frac", type=float, default=None,
                   help="Training fraction (default 0.7).")

    # Modes
    p.add_argument("--plot", action="store_true",
                   help="Generate diagnostic plots for summer idle windows.")
    p.add_argument("--qc-csv", action="store_true",
                   help="Generate a QC summary CSV with per-window RMSE.")
    p.add_argument("--n-plot", type=int, default=None,
                   help="Number of windows to plot (default 3).")
    p.add_argument("--summer-months", type=int, nargs="+", default=None,
                   help="Month numbers for summer selection (default 6 7 8).")

    return p.parse_args()


def _build_config(args: argparse.Namespace) -> UAConfig:
    """Build a UAConfig, applying any CLI overrides."""
    cfg = UAConfig()
    if args.ashp_off is not None:
        cfg.ashp_off_kwh = args.ashp_off
    if args.st_off is not None:
        cfg.st_off_kwh = args.st_off
    if args.imm_off is not None:
        cfg.imm_off_kwh = args.imm_off
    if args.min_idle is not None:
        cfg.min_idle_intervals = args.min_idle
    if args.draw_delta is not None:
        cfg.draw_delta_c = args.draw_delta
    if args.min_windows is not None:
        cfg.min_idle_windows = args.min_windows
    if args.ridge is not None:
        cfg.ridge_alpha = args.ridge
    if args.train_frac is not None:
        cfg.train_frac = args.train_frac
    if args.n_plot is not None:
        cfg.n_plot_windows = args.n_plot
    if args.summer_months is not None:
        cfg.summer_months = args.summer_months
    return cfg


def main() -> int:
    """Run the full UA fitting pipeline.  Returns 0 on success, 1 on warning."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-18s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()
    cfg = _build_config(args)

    # -- 1. Load and clean data -----------------------------------------------
    logger.info("Loading data from %s with mapping %s ...", args.csv, args.yaml)
    df = load_and_clean(args.csv, args.yaml, sampling_minutes=cfg.sampling_minutes)

    # -- 2. Compute solar-thermal energy (needed for ST-off detection) --------
    logger.info("Computing solar-thermal interval energy ...")
    df["st_kwh"] = compute_st_energy(df, dt_minutes=float(cfg.sampling_minutes))
    
    # -- 3. Detect idle windows -----------------------------------------------
    logger.info("Detecting idle windows ...")
    windows, diag_df = detect_idle_windows(df, cfg)

    # Save diagnostics CSV (always, even if too few windows)
    _DIAG_DIR.mkdir(parents=True, exist_ok=True)
    diag_path = _DIAG_DIR / "idle_windows.csv"
    diag_df.to_csv(diag_path, index=False)
    logger.info("Saved idle-window diagnostics to %s", diag_path)

    # -- 4. Check minimum window count ----------------------------------------
    if len(windows) < cfg.min_idle_windows:
        logger.warning(
            "Only %d idle windows found (minimum %d).  "
            "Diagnostics saved but UA fit NOT written.  "
            "Consider relaxing thresholds.",
            len(windows), cfg.min_idle_windows,
        )
        return 1

    # -- 5. Fit UA_loss -------------------------------------------------------
    logger.info("Fitting UA_loss from %d idle windows ...", len(windows))
    result = fit_ua(windows, cfg, output_dir=_OUTPUT_DIR)
    logger.info("Fitted UA_loss = %s", result["UA_loss"])

    # -- 6. Optional: diagnostic plots ----------------------------------------
    if args.plot:
        logger.info("Generating diagnostic plots ...")
        saved = plot_qc(windows, result, cfg, output_dir=_PLOT_DIR)
        logger.info("Saved %d plot(s) to %s", len(saved), _PLOT_DIR)

    # -- 7. Optional: QC CSV --------------------------------------------------
    if args.qc_csv:
        qc_path = _OUTPUT_DIR / "qc_summary.csv"
        import numpy as np
        write_qc_csv(windows, np.array(result["UA_loss"]), cfg, output_path=qc_path)

    logger.info("UA fitting pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
