#!/usr/bin/env python3
"""
DHW_fitting/run_dhw_fitting.py — CLI entrypoint for DHW demand-profile fitting.

Usage
-----
    python DHW_fitting/run_dhw_fitting.py \
        --csv data/FullDS_Findhorn_30min.csv \
        --yaml column_mapping.yaml

Outputs
-------
    DHW_fitting/output/dhw_profile.csv         — (month, slot, mean_V_l, n_events, n_days)
    DHW_fitting/diagnostics/draw_events.csv    — one row per detected event
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure repo root is on sys.path so src.* imports work when run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data_loader import load_and_clean

from DHW_fitting.config import DhwFitConfig
from DHW_fitting import fit_dhw

logger = logging.getLogger("DHW_fitting")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit a monthly DHW demand profile from bottom-node draw events.",
    )
    p.add_argument("--csv", required=True, help="Path to the cleaned CSV.")
    p.add_argument("--yaml", required=True, help="Path to column_mapping.yaml.")
    p.add_argument("--train-frac", type=float, default=None,
                   help="Learning fraction (default 0.7).")
    p.add_argument("--draw-delta", type=float, default=None,
                   help="Draw threshold [°C/interval] (default -0.25).")
    p.add_argument("--t-mains", type=float, default=None,
                   help="Mains cold-water temperature [°C] (default 10.0).")
    p.add_argument("--sampling-minutes", type=int, default=None,
                   help="Data cadence [min] (default 30).")
    p.add_argument("--use-all-data", action="store_true",
                   help="Use the entire dataset for profile learning "
                        "(recommended; the demand profile is an exogenous "
                        "input, not a fitted dynamics parameter).")
    p.add_argument("--min-frequency", type=float, default=None,
                   help="Zero out (month, slot) entries occurring on fewer "
                        "than this fraction of days (default 0.05).")
    p.add_argument("--no-baseline", action="store_true",
                   help="Disable per-month baseline-conduction subtraction.")
    return p.parse_args()


def _build_config(args: argparse.Namespace) -> DhwFitConfig:
    cfg = DhwFitConfig()
    if args.train_frac is not None:
        cfg.train_frac = args.train_frac
    if args.draw_delta is not None:
        cfg.draw_delta_c = args.draw_delta
    if args.t_mains is not None:
        cfg.t_mains_c = args.t_mains
    if args.sampling_minutes is not None:
        cfg.sampling_minutes = args.sampling_minutes
    if args.use_all_data:
        cfg.use_all_data = True
    if args.min_frequency is not None:
        cfg.min_event_frequency = args.min_frequency
    if args.no_baseline:
        cfg.use_baseline_subtraction = False
    return cfg


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-18s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()
    cfg = _build_config(args)

    logger.info("Loading data from %s with mapping %s ...", args.csv, args.yaml)
    df = load_and_clean(args.csv, args.yaml, sampling_minutes=cfg.sampling_minutes)

    summary = fit_dhw.main(cfg, df)

    logger.info("=== DHW profile summary ===")
    for k, v in summary.items():
        logger.info("  %s: %s", k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
