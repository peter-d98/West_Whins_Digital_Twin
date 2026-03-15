#!/usr/bin/env python3
"""
run_ashp_1min.py – ASHP Map Identification from 1-minute Data
==============================================================

Identifies ASHP performance-map coefficients from 1-minute resolution data
using run-based aggregation to isolate clean DHW-only operating intervals.

At 1-minute resolution, mode switches between DHW and space-heating are
detectable: consecutive minutes where the ASHP is running with a rising tank
top temperature and no immersion activity are grouped into "runs", and only
runs ≥ ``--min-run-minutes`` are used for fitting.  This avoids the
space-heating electricity contamination that affects 30-minute back-calculation.

Output
------
``output/params_ashp.json`` — ASHP map coefficients and identification
diagnostics, ready to be consumed by ``run_stage1.py --ashp-params``.

Usage
-----
    python run_ashp_1min.py
    python run_ashp_1min.py --energy-2324 data/my_2324.csv \\
                             --energy-2025 data/my_2025.csv \\
                             --tank data/my_tank.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from src import ashp_model, data_loader, identification

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent

# Reference operating points for map validation (datasheet-anchored)
_REFERENCE_POINTS = [
    ("A-3/W55", -3.0, 55.0),   # expected COP ~2.5–2.8
    ("A7/W55",   7.0, 55.0),   # expected COP ~3.3–3.8
    ("A-3/W35", -3.0, 35.0),   # datasheet COP = 2.91
]


def main(
    energy_2324_path: Path | None = None,
    energy_2025_path: Path | None = None,
    tank_path: Path | None = None,
    yaml_path: Path | None = None,
    output_dir: Path | None = None,
    min_run_minutes: int = 15,
) -> dict:
    """Identify ASHP performance maps from 1-minute data and save results.

    Parameters
    ----------
    energy_2324_path : Path, optional
        Path to the 2023-2024 1-minute energy CSV.
        Default: ``data/Data_WestWhins_2023_2024_1min.csv``.
    energy_2025_path : Path, optional
        Path to the 2025 1-minute energy CSV.
        Default: ``data/Data_WestWhins_2025_final_1min.csv``.
    tank_path : Path, optional
        Path to the 1-minute tank temperature CSV.
        Default: ``data/Data_WestWhins_TankT__with_T_out_1min.csv``.
    yaml_path : Path, optional
        Path to ``column_mapping_1min.yaml``.
        Default: ``column_mapping_1min.yaml`` in the repository root.
    output_dir : Path, optional
        Directory for output files.  Default: ``output/``.
    min_run_minutes : int
        Minimum consecutive qualifying minutes to form a valid DHW run
        (default 15).

    Returns
    -------
    dict
        Summary with run statistics and ASHP map validation values.
    """
    # ---- Default paths ----------------------------------------------------
    energy_2324_path = (
        energy_2324_path or ROOT / "data" / "Data_WestWhins_2023_2024_1min.csv"
    )
    energy_2025_path = (
        energy_2025_path or ROOT / "data" / "Data_WestWhins_2025_final_1min.csv"
    )
    tank_path  = tank_path  or ROOT / "data" / "Data_WestWhins_TankT__with_T_out_1min.csv"
    yaml_path  = yaml_path  or ROOT / "column_mapping_1min.yaml"
    output_dir = output_dir or ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load and merge 1-minute data ----------------------------------
    energy_paths = [p for p in [energy_2324_path, energy_2025_path] if Path(p).exists()]
    if not energy_paths:
        raise FileNotFoundError(
            f"No 1-minute energy CSV files found.  Expected:\n"
            f"  {energy_2324_path}\n  {energy_2025_path}"
        )
    if not Path(tank_path).exists():
        raise FileNotFoundError(f"Tank temperature CSV not found: {tank_path}")

    logger.info("Loading 1-minute data …")
    df = data_loader.load_and_merge_1min(energy_paths, tank_path, yaml_path)

    # Drop rows where all tank temperatures are NaN
    tank_cols = ["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]
    df = df.dropna(subset=tank_cols, how="all")

    # Ensure st_kwh column exists (compute from measured power if not already)
    from src import solar_thermal
    if "st_kwh" not in df.columns:
        df["st_kwh"] = solar_thermal.compute_st_energy(df, dt_minutes=1)

    logger.info(
        "1-min data loaded: %d rows, %s → %s",
        len(df), df.index.min(), df.index.max(),
    )

    # ---- 2. Back-calculate ASHP heat using run-based aggregation ----------
    runs_df = identification.compute_ashp_runs_1min(df, min_run_minutes=min_run_minutes)

    # ---- 3. Log run statistics and back-calculated COP distribution -------
    if len(runs_df) < 20:
        logger.warning(
            "Only %d qualifying runs found (min_run_minutes=%d); "
            "ASHP map may be poorly constrained.  Consider lowering --min-run-minutes.",
            len(runs_df), min_run_minutes,
        )

    if len(runs_df) == 0:
        raise RuntimeError(
            "No qualifying ASHP runs found.  Check that the 1-min data files "
            "contain DHW-mode operating periods."
        )

    lens = runs_df["n_minutes"]
    cops = runs_df["Q_kwh"] / runs_df["P_kwh"].clip(lower=1e-3)  # avoid div-by-zero
    mean_back_cop   = float(cops.mean())
    median_back_cop = float(cops.median())
    run_len_median  = float(lens.median())

    logger.info(
        "Qualifying ASHP runs: %d  |  length min=%d median=%.1f max=%d min",
        len(runs_df), int(lens.min()), run_len_median, int(lens.max()),
    )
    logger.info(
        "Back-calculated COP:  mean=%.2f  median=%.2f",
        mean_back_cop, median_back_cop,
    )

    # ---- 4. Fit ASHP maps from run-level data -----------------------------
    # Convert run totals (kWh) → average power (kW-equivalent) for fitting.
    # fit_ashp_maps expects P_meas_kwh / dt_h → kW; passing dt_h=1.0 and
    # average-kW values (kWh / run_hours) achieves the same result.
    dt_h_runs = runs_df["n_minutes"].values / 60.0
    q_kw_avg  = runs_df["Q_kwh"].values / dt_h_runs
    p_kw_avg  = runs_df["P_kwh"].values / dt_h_runs

    ashp_p = ashp_model.fit_ashp_maps(
        T_out=runs_df["T_out_c"].values,
        T_sink=runs_df["T_sink_c"].values,
        Q_meas_kwh=q_kw_avg,
        P_meas_kwh=p_kw_avg,
        dt_h=1.0,
    )

    # ---- 5. Log map predictions at reference points -----------------------
    cop_ref: dict[str, float] = {}
    logger.info("ASHP map validation at reference operating points:")
    for label, t_out, t_sink in _REFERENCE_POINTS:
        cop_val = float(ashp_model.predict_cop(
            np.array([t_out]), np.array([t_sink]), ashp_p
        )[0])
        cop_ref[f"cop_at_{label}"] = cop_val
        logger.info("  %s (T_out=%.0f°C, T_sink=%.0f°C): COP = %.2f",
                    label, t_out, t_sink, cop_val)

    # ---- 6. Save output/params_ashp.json ----------------------------------
    output_data = {
        "ashp": {
            "a": ashp_p.a.tolist(),
            "b": ashp_p.b.tolist(),
        },
        "identification": {
            "n_runs":                int(len(runs_df)),
            "run_length_median_min": round(run_len_median, 1),
            "mean_back_cop":         round(mean_back_cop, 3),
            "median_back_cop":       round(median_back_cop, 3),
            **{k: round(v, 3) for k, v in cop_ref.items()},
        },
    }

    params_path = output_dir / "params_ashp.json"
    with open(params_path, "w") as fh:
        json.dump(output_data, fh, indent=2)
    logger.info("ASHP params saved to %s", params_path)

    # ---- 7. Return summary dict -------------------------------------------
    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Identify ASHP maps from 1-minute data"
    )
    parser.add_argument(
        "--energy-2324", type=Path, default=None,
        metavar="PATH",
        help="Path to the 2023-2024 1-minute energy CSV.",
    )
    parser.add_argument(
        "--energy-2025", type=Path, default=None,
        metavar="PATH",
        help="Path to the 2025 1-minute energy CSV.",
    )
    parser.add_argument(
        "--tank", type=Path, default=None,
        metavar="PATH",
        help="Path to the 1-minute tank temperature CSV.",
    )
    parser.add_argument(
        "--yaml", type=Path, default=None,
        metavar="PATH",
        help="Path to column_mapping_1min.yaml.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        metavar="DIR",
        help="Output directory (default: output/).",
    )
    parser.add_argument(
        "--min-run-minutes", type=int, default=15,
        metavar="N",
        help="Minimum consecutive DHW-only minutes to form a valid run (default: 15).",
    )
    args = parser.parse_args()
    main(
        energy_2324_path=args.energy_2324,
        energy_2025_path=args.energy_2025,
        tank_path=args.tank,
        yaml_path=args.yaml,
        output_dir=args.output,
        min_run_minutes=args.min_run_minutes,
    )
