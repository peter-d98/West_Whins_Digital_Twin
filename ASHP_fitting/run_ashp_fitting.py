#!/usr/bin/env python3
"""
ASHP_fitting/run_ashp_fitting.py – CLI entrypoint for ASHP performance-map
fitting from 30-minute data.

Usage
-----
    # Basic run (detection + back-calc + fitting, saves JSON and diagnostics):
    python ASHP_fitting/run_ashp_fitting.py \\
        --csv data/FullDS_Findhorn.csv \\
        --yaml column_mapping.yaml

    # With diagnostic plots:
    python ASHP_fitting/run_ashp_fitting.py \\
        --csv data/FullDS_Findhorn.csv \\
        --yaml column_mapping.yaml \\
        --plot

    # Override thresholds:
    python ASHP_fitting/run_ashp_fitting.py \\
        --csv data/FullDS_Findhorn.csv \\
        --yaml column_mapping.yaml \\
        --ashp-off 0.05 --draw-delta -2.0

    # Custom UA fit path:
    python ASHP_fitting/run_ashp_fitting.py \\
        --csv data/FullDS_Findhorn.csv \\
        --yaml column_mapping.yaml \\
        --ua-json path/to/ua_fit.json

All output is written under ASHP_fitting/output/ and ASHP_fitting/diagnostics/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the repository root is on sys.path so that both ``src.*`` and
# ``ASHP_fitting.*`` are importable when running this script directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from src.data_loader import load_and_clean
from src.solar_thermal import compute_st_energy

from ASHP_fitting.config import ASHPFitConfig
from ASHP_fitting.detector import detect_ashp_windows
from ASHP_fitting.evaluate import evaluate_ashp_fit, write_backcalc_csv
from ASHP_fitting.fit_ashp_maps import back_calculate_q_ashp, fit_ashp

logger = logging.getLogger("ASHP_fitting")

# Output directories (relative to ASHP_fitting/)
_ASHP_DIR = Path(__file__).resolve().parent
_OUTPUT_DIR = _ASHP_DIR / "output"
_DIAG_DIR = _ASHP_DIR / "diagnostics"
_PLOT_DIR = _OUTPUT_DIR / "plots"

# Default UA fit path
_DEFAULT_UA_JSON = _REPO_ROOT / "UA_fitting" / "output" / "ua_fit.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit ASHP performance maps from 30-minute data using "
                    "back-calculated condenser heat.",
    )
    p.add_argument("--csv", required=True, help="Path to the cleaned CSV.")
    p.add_argument("--yaml", required=True, help="Path to column_mapping.yaml.")
    p.add_argument("--ua-json", type=str, default=None,
                   help="Path to ua_fit.json (default: UA_fitting/output/ua_fit.json).")

    # Threshold overrides
    p.add_argument("--ashp-off", type=float, default=None,
                   help="ASHP-off threshold [kWh] (default 0.013).")
    p.add_argument("--st-off", type=float, default=None,
                   help="Solar-thermal-off threshold [kWh] (default 0.05).")
    p.add_argument("--imm-off", type=float, default=None,
                   help="Immersion-off threshold [kWh] (default 0.01).")
    p.add_argument("--draw-delta", type=float, default=None,
                   help="Draw delta threshold [°C] (default -1.0).")
    p.add_argument("--min-intervals", type=int, default=None,
                   help="Minimum ASHP-only intervals per window (default 1).")
    p.add_argument("--train-frac", type=float, default=None,
                   help="Training fraction (default 0.7).")
    p.add_argument("--high-load-filter", action="store_true",
                   help="Enable high-load percentile filter for power map.")

    # Modes
    p.add_argument("--plot", action="store_true",
                   help="Generate diagnostic plots for longest ASHP windows.")
    p.add_argument("--n-plot", type=int, default=None,
                   help="Number of windows to plot (default 5).")

    return p.parse_args()


def _build_config(args: argparse.Namespace) -> ASHPFitConfig:
    """Build an ASHPFitConfig, applying any CLI overrides."""
    cfg = ASHPFitConfig()
    if args.ashp_off is not None:
        cfg.ashp_off_kwh = args.ashp_off
    if args.st_off is not None:
        cfg.st_off_kwh = args.st_off
    if args.imm_off is not None:
        cfg.imm_off_kwh = args.imm_off
    if args.draw_delta is not None:
        cfg.draw_delta_c = args.draw_delta
    if args.min_intervals is not None:
        cfg.min_ashp_intervals = args.min_intervals
    if args.train_frac is not None:
        cfg.train_frac = args.train_frac
    if args.high_load_filter:
        cfg.apply_high_load_filter = True
    if args.n_plot is not None:
        cfg.n_plot_windows = args.n_plot
    return cfg


def _load_ua_priors(ua_json_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load UA_loss and UA_adj priors from ua_fit.json.  Fail loudly if missing.

    Returns
    -------
    ua_loss : np.ndarray, shape (4,)
    ua_adj  : np.ndarray, shape (3,)  — may be zeros if key absent
    """
    if not ua_json_path.exists():
        raise FileNotFoundError(
            f"UA priors file not found: {ua_json_path}\n"
            f"Run the UA_fitting pipeline first:\n"
            f"  python UA_fitting/run_ua_fitting.py --csv <csv> --yaml <yaml>\n"
            f"Then re-run this ASHP fitting pipeline."
        )
    with open(ua_json_path, "r", encoding="utf-8") as f:
        ua_data = json.load(f)

    ua_loss = np.array(ua_data["UA_loss"], dtype=float)
    if ua_loss.shape != (4,):
        raise ValueError(
            f"Expected UA_loss array of shape (4,), got {ua_loss.shape} "
            f"from {ua_json_path}"
        )

    ua_adj = np.array(ua_data.get("UA_adj", [0.0, 0.0, 0.0]), dtype=float)
    if ua_adj.shape != (3,):
        raise ValueError(
            f"Expected UA_adj array of shape (3,), got {ua_adj.shape} "
            f"from {ua_json_path}"
        )

    logger.info("Loaded UA_loss priors: %s from %s", ua_loss.tolist(), ua_json_path)
    logger.info("Loaded UA_adj priors:  %s from %s", ua_adj.tolist(), ua_json_path)
    return ua_loss, ua_adj


def main() -> int:
    """Run the full ASHP fitting pipeline.  Returns 0 on success, 1 on failure."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-18s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()
    cfg = _build_config(args)

    # -- 1. Load UA priors (fail early if missing) ---------------------------
    ua_json_path = Path(args.ua_json) if args.ua_json else _DEFAULT_UA_JSON
    try:
        ua_loss, ua_adj = _load_ua_priors(ua_json_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    # -- 2. Load and clean data ----------------------------------------------
    logger.info("Loading data from %s with mapping %s ...", args.csv, args.yaml)
    df = load_and_clean(args.csv, args.yaml, sampling_minutes=cfg.sampling_minutes)

    # -- 3. Compute solar-thermal energy (needed for ST-off detection) -------
    logger.info("Computing solar-thermal interval energy ...")
    df["st_kwh"] = compute_st_energy(df, dt_minutes=float(cfg.sampling_minutes))

    # -- 4. Detect ASHP-only windows -----------------------------------------
    logger.info("Detecting ASHP-only windows ...")
    windows, diag_df = detect_ashp_windows(df, cfg)

    # Save diagnostics CSV (always)
    _DIAG_DIR.mkdir(parents=True, exist_ok=True)
    diag_path = _DIAG_DIR / "ashp_intervals.csv"
    diag_df.to_csv(diag_path, index=False)
    logger.info("Saved ASHP interval diagnostics to %s", diag_path)

    if not windows:
        logger.error("No ASHP-only windows found.  Check thresholds and data.")
        return 1

    # -- 5. Fit ASHP maps (back-calc + fitting) ------------------------------
    logger.info("Fitting ASHP maps from %d windows ...", len(windows))
    result = fit_ashp(df, windows, ua_loss, cfg, ua_adj=ua_adj, output_dir=_OUTPUT_DIR)

    if "error" in result.get("identification", {}):
        logger.error("ASHP fitting failed: %s", result["identification"]["error"])
        return 1

    logger.info("Fitted ASHP COP coefficients (c): %s", result["ashp"]["c"])

    # -- 6. Save back-calculation details CSV --------------------------------
    n_train = int(len(df) * cfg.train_frac)
    df_train = df.iloc[:n_train]
    bc_df = back_calculate_q_ashp(df_train, windows, ua_loss, cfg, ua_adj=ua_adj)
    bc_path = _DIAG_DIR / "backcalc_details.csv"
    write_backcalc_csv(bc_df, output_path=bc_path)

    # -- 7. Optional: diagnostic plots ---------------------------------------
    if args.plot:
        logger.info("Generating diagnostic plots ...")
        saved = evaluate_ashp_fit(df, windows, result, cfg, output_dir=_PLOT_DIR)
        logger.info("Saved %d plot(s) to %s", len(saved), _PLOT_DIR)

    logger.info("ASHP fitting pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
