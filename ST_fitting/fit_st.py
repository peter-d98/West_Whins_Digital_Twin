"""
ST_fitting.fit_st – Compute empirical f_st from ST-only windows.

This module computes the solar-thermal heat-distribution vector f_st in
the same way that ASHP_fitting.fit_ashp_maps computes f_ashp: by measuring
how much each node's stored energy changed over clean ST-only windows and
normalising to a distribution that sums to 1.

Physics
-------
For each accepted ST-only window, the temperature difference between the
start (T[first_k - 1]) and end (T[last_k]) captures where heat genuinely
settled after a solar-thermal charging episode.  Unlike the ASHP case,
*all four nodes* are included because the ST coil is in the bottom node
and heat spreads upward — we expect the bottom node to receive the
largest share.

For each window:

    dT_i = T_i[last_k] - T_i[first_k - 1]         (°C)
    energy_i = NODE_CAP_i × dT_i                    (kJ)

Cooling artefacts (energy_i < 0) are clipped to zero.  The per-window
distribution is:

    f_st_window_i = energy_i / Σ_j energy_j

The overall f_st is the median across all valid windows (median is more
robust to outlier windows than the mean).

Diagnostics
-----------
A per-window CSV is written with start/end times, interval count, total
ST energy delivered (measured), total stored energy change, and per-node
distributions for manual inspection.

NODE_CAP values
---------------
The same per-node thermal capacities as the tank model are used:
  - Bottom node : 170 L  → 711.62 kJ/K
  - Mid, Mid-Hi, Top : 380/3 ≈ 126.67 L each → 530.21 kJ/K each
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ST_fitting.config import STFitConfig
from ST_fitting.detector import STWindow
from src.tank_model import NODE_CAP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def fit_f_st(
    df: pd.DataFrame,
    windows: List[STWindow],
    cfg: Optional[STFitConfig] = None,
    *,
    output_dir: Optional[Path] = None,
) -> Dict:
    """Compute empirical f_st from the temperature response during ST-only windows.

    Parameters
    ----------
    df : pd.DataFrame
        Full cleaned DataFrame (must include ``st_power_kw`` column).
    windows : list[STWindow]
        Accepted ST-only windows from the detector.
    cfg : STFitConfig, optional
    output_dir : Path, optional
        Directory for ``st_fit.json``.  Created if needed.

    Returns
    -------
    result : dict
        ``{"f_st": [...], "identification": {...}}``
    """
    if cfg is None:
        cfg = STFitConfig()

    # -- Step 1: Extract training portion (same slice as detector) -----------
    n_train = int(len(df) * cfg.train_frac)
    df_train = df.iloc[:n_train]

    T_train_vals = df_train[cfg.node_cols].values   # (N_train, 4)
    dt_h = cfg.sampling_minutes / 60.0
    Q_st_vals = (df_train[cfg.st_power_col].fillna(0.0) * dt_h).values  # kWh/interval

    if not windows:
        logger.error("No ST-only windows available for f_st estimation.")
        return _empty_result()

    # -- Step 2: Compute per-window energy distribution ----------------------
    # For each window, compute the net temperature rise across all 4 nodes
    # and convert to energy using the per-node thermal capacities.
    window_records = []
    f_st_rows = []

    for w in windows:
        first_k = w.indices[0]
        last_k = w.indices[-1]

        # Temperature change over the full window.
        # T[first_k - 1] is the pre-window state (before any ST heat).
        # T[last_k] is the state at the end of the window.
        if first_k == 0:
            # Cannot compute dT without a prior row; skip this window.
            logger.debug(
                "Skipping window %d: starts at index 0 (no prior row).",
                w.window_id,
            )
            continue

        dT = T_train_vals[last_k] - T_train_vals[first_k - 1]   # shape (4,)
        energy = NODE_CAP * dT    # kJ per node

        # Clip negative (cooling) artefacts — a node that cooled during a
        # ST-only window lost heat to ambient or neighbours, not from ST input.
        energy = np.maximum(energy, 0.0)
        total_energy = energy.sum()

        # Total measured ST energy delivered during this window [kWh]
        q_st_window_kwh = float(Q_st_vals[w.indices].sum())

        # Per-node fraction for this window
        if total_energy > 1e-3:
            f_row = energy / total_energy
            f_st_rows.append(f_row)
        else:
            f_row = np.full(4, np.nan)

        window_records.append({
            "window_id": w.window_id,
            "start": w.start,
            "end": w.end,
            "n_intervals": w.n_intervals,
            "Q_st_meas_kwh": round(q_st_window_kwh, 6),
            "dE_stored_kWh": round(float(total_energy/3600), 2),
            "Q_meas:Q_stored_ratio": round(float(q_st_window_kwh / (total_energy/3600)), 4) if total_energy > 1e-3 else None,
            "dT_bottom": round(float(dT[0]), 4),
            "dT_mid": round(float(dT[1]), 4),
            "dT_mid_hi": round(float(dT[2]), 4),
            "dT_top": round(float(dT[3]), 4),
            "f_bottom": round(float(f_row[0]), 4),
            "f_mid": round(float(f_row[1]), 4),
            "f_mid_hi": round(float(f_row[2]), 4),
            "f_top": round(float(f_row[3]), 4),
        })

    window_df = pd.DataFrame(window_records)

    if not f_st_rows:
        logger.error(
            "No valid windows produced positive energy change.  "
            "Cannot estimate f_st."
        )
        return _empty_result()

    # -- Step 3: Aggregate across windows ------------------------------------
    f_st_arr = np.array(f_st_rows)                # shape (n_valid, 4)
    f_st_median = np.median(f_st_arr, axis=0)
    f_st_mean = np.mean(f_st_arr, axis=0)
    f_st_std = np.std(f_st_arr, axis=0)

    # Normalise the median so it sums to exactly 1
    _s = f_st_median.sum()
    if _s > 0:
        f_st_median /= _s

    logger.info(
        "Empirical f_st (median): bottom=%.3f mid=%.3f mid-hi=%.3f top=%.3f",
        *f_st_median,
    )
    logger.info(
        "Empirical f_st (mean):   bottom=%.3f mid=%.3f mid-hi=%.3f top=%.3f",
        *f_st_mean,
    )

    # -- Step 4: Summary statistics ------------------------------------------
    n_valid = len(f_st_rows)
    n_total = len(windows)
    total_st_kwh = float(window_df["Q_st_meas_kwh"].sum())
    mean_st_per_window = float(window_df["Q_st_meas_kwh"].mean())
    mean_intervals = float(window_df["n_intervals"].mean())

    # -- Step 5: Build result dictionary -------------------------------------
    result = {
        "f_st": [round(float(v), 4) for v in f_st_median],
        "identification": {
            "n_windows_total": n_total,
            "n_windows_valid": n_valid,
            "n_windows_skipped": n_total - n_valid,
            "total_st_energy_kwh": round(total_st_kwh, 3),
            "mean_st_per_window_kwh": round(mean_st_per_window, 4),
            "mean_intervals_per_window": round(mean_intervals, 2),
            "f_st_stats": {
                "nodes": ["bottom", "mid", "mid_hi", "top"],
                "mean": [round(float(v), 4) for v in f_st_mean],
                "median": [round(float(v), 4) for v in f_st_median],
                "std": [round(float(v), 4) for v in f_st_std],
                "n_windows": n_valid,
            },
            "thresholds": {
                "st_flow_dt_min_c": cfg.st_flow_dt_min_c,
                "st_flow_min_l": cfg.st_flow_min_l,
                "st_power_min_kw": cfg.st_power_min_kw,
                "ashp_off_kwh": cfg.ashp_off_kwh,
                "imm_off_kwh": cfg.imm_off_kwh,
                "min_st_intervals": cfg.min_st_intervals,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    # -- Step 6: Save outputs to disk ----------------------------------------
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON with f_st and metadata
        out_json = output_dir / "st_fit.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info("Saved ST fit to %s", out_json)

    return result, window_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_result() -> tuple[Dict, pd.DataFrame]:
    """Return an empty result when fitting cannot proceed."""
    return (
        {
            "f_st": [0.0, 0.0, 0.0, 0.0],
            "identification": {"error": "insufficient_data"},
        },
        pd.DataFrame(),
    )
