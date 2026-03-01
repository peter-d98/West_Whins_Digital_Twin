#!/usr/bin/env python3
"""
run_stage1.py – Stage-1 Digital Twin Pipeline
==============================================

Orchestrates data loading, parameter identification, and evaluation
for the West Whins DHW system grey-box model.

Usage
-----
    python run_stage1.py                          # defaults
    python run_stage1.py --csv path/to/data.csv   # custom paths
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src import data_loader, identification, evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent


def main(
    csv_path: Path | None = None,
    yaml_path: Path | None = None,
    output_dir: Path | None = None,
    train_frac: float = 0.7,
    max_nfev: int = 300,
) -> dict:
    csv_path  = csv_path  or ROOT / "FullDS_Findhorn.csv"
    yaml_path = yaml_path or ROOT / "column_mapping.yaml"
    output_dir = output_dir or ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load & clean data ------------------------------------------------
    logger.info("Loading data from %s", csv_path)
    df = data_loader.load_and_clean(csv_path, yaml_path)

    # Drop rows where all tank temperatures are NaN
    tank_cols = ["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]
    df = df.dropna(subset=tank_cols, how="all")
    logger.info("After dropping all-NaN tank rows: %d rows", len(df))

    # ---- Node-ordering diagnostic -----------------------------------------
    ordering = data_loader.node_ordering_check(df)
    logger.info("Node ordering satisfied: %.1f %%", ordering.mean() * 100)

    # ---- Identification ---------------------------------------------------
    logger.info("Running identification (train_frac=%.2f) …", train_frac)
    id_result, df_train, df_val = identification.run_identification(
        df, train_frac=train_frac, max_nfev=max_nfev,
    )

    # ---- Save parameters --------------------------------------------------
    params_file = output_dir / "params.json"
    _save_params(id_result, params_file)
    logger.info("Parameters saved to %s", params_file)

    # ---- Evaluate ---------------------------------------------------------
    plot_dir = output_dir / "plots"
    summary_train = evaluation.evaluate(
        df_train, id_result, label="train", plot_dir=plot_dir,
    )
    summary_val = evaluation.evaluate(
        df_val, id_result, label="validation", plot_dir=plot_dir,
    )

    summary = {"train": summary_train, "val": summary_val}
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    logger.info("Summary saved to %s", summary_file)

    return summary


def _save_params(id_result: identification.IdentificationResult, path: Path) -> None:
    """Serialise identified parameters to JSON."""
    data = {
        "tank": {
            "UA_loss": id_result.tank_params.UA_loss.tolist(),
            "UA_adj":  id_result.tank_params.UA_adj.tolist(),
            "f_st":    id_result.tank_params.f_st.tolist(),
            "f_ashp":  id_result.tank_params.f_ashp.tolist(),
            "f_imm":   id_result.tank_params.f_imm.tolist(),
            "mix_coeff": id_result.tank_params.mix_coeff,
        },
        "ashp": {
            "a": id_result.ashp_params.a.tolist(),
            "b": id_result.ashp_params.b.tolist(),
        },
        "hx_effectiveness": id_result.hx_effectiveness,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


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
    parser = argparse.ArgumentParser(description="Stage-1 DHW Digital Twin")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--yaml", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--max-nfev", type=int, default=300)
    args = parser.parse_args()
    main(args.csv, args.yaml, args.output, args.train_frac, args.max_nfev)
