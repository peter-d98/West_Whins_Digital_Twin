#!/usr/bin/env python3
"""
Global_fitting/run_global_fitting.py – CLI entrypoint for global tank-parameter
fitting using frozen UA_loss and ASHP priors.

Usage
-----
    # Basic run:
    python Global_fitting/run_global_fitting.py \\
        --csv data/FullDS_Findhorn.csv \\
        --yaml column_mapping.yaml

    # With diagnostic plots:
    python Global_fitting/run_global_fitting.py \\
        --csv data/FullDS_Findhorn.csv \\
        --yaml column_mapping.yaml \\
        --plot

    # Custom prior paths:
    python Global_fitting/run_global_fitting.py \\
        --csv data/FullDS_Findhorn.csv \\
        --yaml column_mapping.yaml \\
        --ua-json UA_fitting/output/ua_fit.json \\
        --ashp-json ASHP_fitting/output/ashp_fit.json

All output is written under Global_fitting/output/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Ensure the repository root is on sys.path so that both ``src.*`` and
# ``Global_fitting.*`` are importable when running this script directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data_loader import load_and_clean
from src.solar_thermal import compute_st_energy

from Global_fitting.config import GlobalFitConfig
from Global_fitting.evaluate import evaluate_split, plot_prediction_errors
from Global_fitting.fit_global import (
    _load_ashp_priors,
    _load_ua_priors,
    run_global_fit,
)

logger = logging.getLogger("Global_fitting")

# Output directories (relative to Global_fitting/)
_GLOBAL_DIR = Path(__file__).resolve().parent
_OUTPUT_DIR = _GLOBAL_DIR / "output"
_PLOT_DIR = _OUTPUT_DIR / "plots"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit global tank parameters (UA_adj + optionally UA_loss[0]) "
                    "using frozen UA and ASHP priors.",
    )
    p.add_argument("--csv", required=True, help="Path to the cleaned CSV (e.g. data/FullDS_Findhorn_5min.csv).")
    p.add_argument("--yaml", required=True, help="Path to column_mapping YAML (e.g. column_mapping_5min.yaml).")
    p.add_argument("--ua-json", type=str, default=None,
                   help="Path to ua_fit.json (default: UA_fitting/output/ua_fit.json).")
    p.add_argument("--ashp-json", type=str, default=None,
                   help="Path to ashp_fit.json (default: ASHP_fitting/output/ashp_fit.json).")
    p.add_argument("--train-frac", type=float, default=None,
                   help="Training fraction (default 0.7).")
    p.add_argument("--max-nfev", type=int, default=None,
                   help="Max function evaluations (default 500).")
    p.add_argument("--freeze-ua-loss-bottom", action="store_true",
                   help="Freeze UA_loss[0] at its prior value (default: free).")
    p.add_argument("--plot", action="store_true",
                   help="Generate diagnostic plots of prediction errors.")
    return p.parse_args()


def _build_config(args: argparse.Namespace) -> GlobalFitConfig:
    """Build a GlobalFitConfig, applying any CLI overrides."""
    cfg = GlobalFitConfig()
    cfg.data_csv = Path(args.csv)
    cfg.column_mapping_yaml = Path(args.yaml)
    if args.ua_json is not None:
        cfg.ua_fit_path = Path(args.ua_json)
    if args.ashp_json is not None:
        cfg.ashp_fit_path = Path(args.ashp_json)
    if args.train_frac is not None:
        cfg.train_frac = args.train_frac
    if args.max_nfev is not None:
        cfg.max_nfev = args.max_nfev
    if args.freeze_ua_loss_bottom:
        cfg.free_ua_loss_bottom = False
    return cfg


def _json_default(obj):
    """JSON serialiser fallback for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main() -> int:
    """Run the full global fitting pipeline.  Returns 0 on success, 1 on failure."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-18s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()
    cfg = _build_config(args)

    # -- 1. Validate priors exist (fail early) --------------------------------
    try:
        ua_loss, ua_adj = _load_ua_priors(cfg.ua_fit_path)
        ashp_params = _load_ashp_priors(cfg.ashp_fit_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    # -- 2. Run fitting -------------------------------------------------------
    logger.info("Running global fitting ...")
    result = run_global_fit(cfg)

    logger.info("Fitted UA_adj: %s", result.tank_params.UA_adj.tolist())
    logger.info("Final UA_loss: %s", result.tank_params.UA_loss.tolist())
    logger.info("Fixed f_ashp (from ashp_fit.json): %s", result.tank_params.f_ashp.tolist())

    # -- 3. Evaluate on train and validation ----------------------------------
    logger.info("Loading data for evaluation ...")
    df = load_and_clean(cfg.data_csv, cfg.column_mapping_yaml,
                        sampling_minutes=cfg.sampling_minutes)
    df["st_kwh"] = compute_st_energy(df, dt_minutes=float(cfg.sampling_minutes))
    tank_cols = ["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]
    df = df.dropna(subset=tank_cols, how="all")

    split_idx = int(len(df) * cfg.train_frac)
    df_train = df.iloc[:split_idx]
    df_val = df.iloc[split_idx:]

    summary_train = evaluate_split(df_train, result.tank_params, ashp_params, "train",
                                    ua_loss=ua_loss, sampling_minutes=cfg.sampling_minutes,
                                    ua_adj=ua_adj)
    summary_val = evaluate_split(df_val, result.tank_params, ashp_params, "validation",
                                  ua_loss=ua_loss, sampling_minutes=cfg.sampling_minutes,
                                  ua_adj=ua_adj)

    # -- 4. Save output JSON --------------------------------------------------
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_data = {
        "UA_loss": result.tank_params.UA_loss.tolist(),
        "UA_adj": result.tank_params.UA_adj.tolist(),
        "f_st": result.tank_params.f_st.tolist(),
        "f_ashp": result.tank_params.f_ashp.tolist(),
        "f_imm": result.tank_params.f_imm.tolist(),
        "identification": {
            "train_rmse": summary_train["node_rmse"],
            "val_rmse": summary_val["node_rmse"],
            "train_mae": summary_train["node_mae"],
            "val_mae": summary_val["node_mae"],
            "n_train_intervals": summary_train["n_intervals"],
            "n_val_intervals": summary_val["n_intervals"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    output_path = _OUTPUT_DIR / "global_fit.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, default=_json_default)
    logger.info("Saved fitted parameters to %s", output_path)

    # -- 5. Optional: diagnostic plots ----------------------------------------
    if args.plot:
        logger.info("Generating diagnostic plots ...")
        plot_prediction_errors(
            df_train, result.tank_params, ashp_params, "train", _PLOT_DIR,
            ua_loss=ua_loss, sampling_minutes=cfg.sampling_minutes, ua_adj=ua_adj,
        )
        plot_prediction_errors(
            df_val, result.tank_params, ashp_params, "validation", _PLOT_DIR,
            ua_loss=ua_loss, sampling_minutes=cfg.sampling_minutes, ua_adj=ua_adj,
        )

    logger.info("Global fitting pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
