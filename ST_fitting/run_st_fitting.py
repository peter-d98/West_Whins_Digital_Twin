#!/usr/bin/env python3
"""
ST_fitting/run_st_fitting.py – CLI entrypoint for solar-thermal
heat-distribution (f_st) identification from 5-minute data.

Usage
-----
    # Basic run (detection + f_st estimation, saves JSON and diagnostics):
    python ST_fitting/run_st_fitting.py \\
        --csv data/FullDS_Findhorn_5min.csv \\
        --yaml column_mapping_5min.yaml

    # With diagnostic plot of the longest ST window:
    python ST_fitting/run_st_fitting.py \\
        --csv data/FullDS_Findhorn_5min.csv \\
        --yaml column_mapping_5min.yaml \\
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
import json
import logging
import sys
from pathlib import Path

# Ensure the repository root is on sys.path so that both ``src.*`` and
# ``ST_fitting.*`` are importable when running this script directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from src.data_loader import load_and_clean
from src.solar_thermal import compute_st_energy
from src.tank_model import TankParams

from ST_fitting.config import STFitConfig
from ST_fitting.detector import detect_st_windows
from ST_fitting.evaluate import plot_st_window, write_interval_csv, write_window_csv
from ST_fitting.fit_st import fit_f_st

logger = logging.getLogger("ST_fitting")

# Output directories (relative to ST_fitting/)
_ST_DIR = Path(__file__).resolve().parent
_OUTPUT_DIR = _ST_DIR / "output"
_DIAG_DIR = _ST_DIR / "diagnostics"
_PLOT_DIR = _OUTPUT_DIR / "plots"

# Default global_fit.json for tank params (needed for simulation plot)
_DEFAULT_GLOBAL_JSON = _REPO_ROOT / "Global_fitting" / "output" / "global_fit.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute empirical f_st (solar-thermal heat distribution) "
                    "from clean ST-only windows in 5-minute data.",
    )
    p.add_argument("--csv", required=True, help="Path to the cleaned CSV.")
    p.add_argument("--yaml", required=True, help="Path to column_mapping.yaml.")
    p.add_argument("--global-json", type=str, default=None,
                   help="Path to global_fit.json for tank params "
                        "(default: Global_fitting/output/global_fit.json).")

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
                   help="Generate diagnostic plot for the longest ST window.")
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


def _load_tank_params(path: Path) -> TankParams:
    """Load TankParams from global_fit.json for use in simulation plots.

    Uses the f_st from the JSON as-is (it will be the old hard-coded value);
    the plot compares the current model prediction against measurements to
    help evaluate the newly computed f_st.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"global_fit.json not found at {path}\n"
            f"Run the Global_fitting pipeline first, or use --global-json."
        )
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    p = TankParams()
    p.UA_loss = np.array(d["UA_loss"], dtype=float)
    p.UA_adj = np.array(d["UA_adj"], dtype=float)
    p.f_st = np.array(d["f_st"], dtype=float)
    p.f_ashp = np.array(d["f_ashp"], dtype=float)
    p.f_imm = np.array(d["f_imm"], dtype=float)
    return p


def main() -> int:
    """Run the full ST fitting pipeline.  Returns 0 on success, 1 on failure."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-18s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()
    cfg = _build_config(args)

    # -- 1. Load and clean data ----------------------------------------------
    logger.info("Loading data from %s with mapping %s ...", args.csv, args.yaml)
    df = load_and_clean(args.csv, args.yaml, sampling_minutes=cfg.sampling_minutes)

    # -- 2. Compute solar-thermal energy -------------------------------------
    logger.info("Computing solar-thermal interval energy ...")
    df["st_kwh"] = compute_st_energy(df, dt_minutes=float(cfg.sampling_minutes))

    # -- 3. Detect ST-only windows -------------------------------------------
    logger.info("Detecting ST-only windows ...")
    windows, diag_df = detect_st_windows(df, cfg)

    # Save interval diagnostics CSV (always)
    _DIAG_DIR.mkdir(parents=True, exist_ok=True)
    diag_path = _DIAG_DIR / "st_intervals.csv"
    write_interval_csv(diag_df, output_path=diag_path)

    if not windows:
        logger.error("No ST-only windows found.  Check thresholds and data.")
        return 1

    # -- 4. Compute f_st from ST-only windows --------------------------------
    logger.info("Computing f_st from %d windows ...", len(windows))
    result, window_df = fit_f_st(df, windows, cfg, output_dir=_OUTPUT_DIR)

    if "error" in result.get("identification", {}):
        logger.error("ST fitting failed: %s", result["identification"]["error"])
        return 1

    logger.info("Computed f_st: %s", result["f_st"])

    # -- 5. Save window-level diagnostics CSV --------------------------------
    window_csv_path = _DIAG_DIR / "st_windows.csv"
    write_window_csv(window_df, output_path=window_csv_path)

    # -- 6. Optional: diagnostic plot ----------------------------------------
    if args.plot:
        global_json = Path(args.global_json) if args.global_json else _DEFAULT_GLOBAL_JSON
        try:
            tank_params = _load_tank_params(global_json)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Cannot generate plot: %s", exc)
            logger.info("ST fitting complete (plots skipped).")
            return 0

        # Select the longest window(s) for plotting
        sorted_windows = sorted(windows, key=lambda w: w.n_intervals, reverse=True)
        n_plot = min(cfg.n_plot_windows, len(sorted_windows))
        _PLOT_DIR.mkdir(parents=True, exist_ok=True)

        for w in sorted_windows[:n_plot]:
            plot_st_window(df, w, tank_params, cfg, output_dir=_PLOT_DIR)

        logger.info("Saved %d plot(s) to %s", n_plot, _PLOT_DIR)

    logger.info("ST fitting pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
