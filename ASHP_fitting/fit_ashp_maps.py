"""
ASHP_fitting.fit_ashp_maps – Back-calculate ASHP heat and fit performance maps.

This module contains:
  1. ``back_calculate_q_ashp`` — a local back-calculation function that
     computes condenser heat delivery from the 4-node energy balance using
     measured tank temperatures and UA_loss priors from ``ua_fit.json``.
  2. ``fit_ashp`` — the main entry point that back-calculates Q, filters
     Q > 0, and delegates to ``src.ashp_model.fit_ashp_maps`` for the
     bilinear map fitting.

Back-calculation physics
------------------------
For each accepted ASHP-only interval *k*:

    Q_ashp_kJ = Σ_i NODE_CAP_i × (T_i[k] − T_i[k−1])
                + Σ_i UA_loss_i × (T_i[k−1] − T_amb[k]) × dt_s

    Q_ashp_kWh = Q_ashp_kJ / 3600

NODE_CAP is per-node here (not the uniform value from tank_model) because
the ASHP HX draws from the mid node and returns to the top node — the bottom
node is not directly charged.  The tank geometry used here is:
  - Bottom node : 170 L  (NODE_CAP = 711.62 kJ/K)
  - Mid, Mid-Hi, Top : 380 L split equally → 126.67 L each
                       (NODE_CAP = 530.21 kJ/K each)

Units
-----
- NODE_CAP_ASHP : array [kJ/K], shape (4,), bottom→top
- dt_s          : seconds (1800 for 30-min)
- UA_loss       : kW/K — so UA_loss × ΔT × dt_s gives kJ
- Q_ashp        : kWh per interval
- P_meas        : kWh per interval (measured ASHP electricity)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ASHP_fitting.config import ASHPFitConfig
from ASHP_fitting.detector import ASHPWindow

# Read-only imports from the main codebase
from src.ashp_model import (
    ASHPParams,
    fit_ashp_maps,
    predict_cop,
    sink_proxy,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-node thermal capacities for this ASHP fitting pipeline
# ---------------------------------------------------------------------------
# The ASHP HX draws from the mid node and returns heated water to the top
# node, so the bottom node is never directly charged.  The physical geometry
# differs from the equal-split assumed by src.tank_model.NODE_CAP:
#   Bottom node : 170 L
#   Mid, Mid-Hi, Top : 380 L split equally → 126.67 L each
# NODE_CAP_i = volume_i [L] × RHO [kg/L] × CP [kJ/(kg·K)]
_RHO = 1.0          # kg/L  (water density)
_CP  = 4.186        # kJ/(kg·K)
NODE_CAP_ASHP = np.array([
    170.0          * _RHO * _CP,   # bottom  → 711.62 kJ/K
    (380.0 / 3.0)  * _RHO * _CP,   # mid     → 530.21 kJ/K
    (380.0 / 3.0)  * _RHO * _CP,   # mid-hi  → 530.21 kJ/K
    (380.0 / 3.0)  * _RHO * _CP,   # top     → 530.21 kJ/K
])  # shape (4,), kJ/K, bottom→top


# ---------------------------------------------------------------------------
# Local back-calculation (self-contained, does not import from identification)
# ---------------------------------------------------------------------------

def back_calculate_q_ashp(
    df_train: pd.DataFrame,
    windows: List[ASHPWindow],
    ua_loss: np.ndarray,
    cfg: ASHPFitConfig,
) -> pd.DataFrame:
    """Back-calculate ASHP condenser heat for every interval in *windows*.

    Parameters
    ----------
    df_train : pd.DataFrame
        The training slice of the cleaned dataset (same that was passed to
        the detector).  Indexed by datetime; positional indices in
        ``ASHPWindow.indices`` refer to ``df_train.iloc``.
    windows : list[ASHPWindow]
        Accepted ASHP-only windows from the detector.
    ua_loss : array of 4 UA_loss values [kW/K] (bottom→top).
    cfg : ASHPFitConfig

    Returns
    -------
    pd.DataFrame
        One row per accepted ASHP-only interval.  Columns:
        ``time``, ``Q_back_kwh``, ``P_meas_kwh``, ``T_out_c``, ``T_sink_c``,
        ``window_id``.
    """
    dt_s = cfg.sampling_minutes * 60.0
    T = df_train[cfg.node_cols].values          # (N_train, 4)
    T_amb = df_train[cfg.t_amb_col].fillna(
        df_train[cfg.t_amb_col].median()
    ).values
    T_out = df_train[cfg.t_out_col].fillna(
        df_train[cfg.t_out_col].median()
    ).values
    P_ashp = df_train["ashp_inst_kwh"].fillna(0.0).values

    records = []
    for w in windows:
        for k in w.indices:
            # k is a positional (iloc) index; k-1 is the previous interval.
            # Only nodes 1–3 (mid, mid-hi, top) are included: the bottom node
            # is never directly charged by the ASHP HX circuit, its temperature
            # rise during charging is conduction-driven, and its fitted UA_loss
            # is negative (mixing artefact) — all of which inflate Q if included.
            storage_kJ = 0.0
            loss_sum = 0.0
            for i in range(1, 4):
                storage_kJ += NODE_CAP_ASHP[i] * (T[k, i] - T[k - 1, i])
                loss_sum += ua_loss[i] * (T[k - 1, i] - T_amb[k]) * dt_s

            Q_kJ = storage_kJ + loss_sum
            Q_kwh = Q_kJ / 3600.0

            t_sink = float(sink_proxy(T[k, 1], T[k, 3]))  # mid, top

            records.append({
                "time": df_train.index[k],
                "Q_back_kwh": Q_kwh,
                "P_meas_kwh": float(P_ashp[k]),
                "T_out_c": float(T_out[k]),
                "T_sink_c": t_sink,
                "window_id": w.window_id,
            })

    result_df = pd.DataFrame(records)
    logger.info("Back-calculated Q for %d intervals across %d windows.",
                len(result_df), len(windows))
    return result_df


# ---------------------------------------------------------------------------
# Main fitting entry point
# ---------------------------------------------------------------------------

def fit_ashp(
    df: pd.DataFrame,
    windows: List[ASHPWindow],
    ua_loss: np.ndarray,
    cfg: Optional[ASHPFitConfig] = None,
    *,
    output_dir: Optional[Path] = None,
) -> Dict:
    """Back-calculate ASHP heat, filter Q > 0, and fit bilinear maps.

    Parameters
    ----------
    df : pd.DataFrame
        Full cleaned DataFrame.
    windows : list[ASHPWindow]
        Accepted ASHP-only windows from the detector.
    ua_loss : array of shape (4,) — UA_loss priors [kW/K].
    cfg : ASHPFitConfig, optional
    output_dir : Path, optional
        Directory for ``ashp_fit.json``.  Created if needed.

    Returns
    -------
    result : dict
        ``{"ashp": {"a": [...], "b": [...]}, "identification": {...}}``
    """
    if cfg is None:
        cfg = ASHPFitConfig()

    dt_h = cfg.sampling_minutes / 60.0

    # -- Step 1: Extract training portion (same slice as detector) -----------
    n_train = int(len(df) * cfg.train_frac)
    df_train = df.iloc[:n_train]

    # -- Step 2: Back-calculate Q_ashp for all accepted intervals ------------
    bc_df = back_calculate_q_ashp(df_train, windows, ua_loss, cfg)

    if bc_df.empty:
        logger.error("No ASHP-only intervals available for back-calculation.")
        return _empty_result()

    # -- Step 3: Filter Q_back > 0 (strictly positive) ----------------------
    n_total = len(bc_df)
    bc_pos = bc_df[bc_df["Q_back_kwh"] > 0.0].copy()
    n_pos = len(bc_pos)
    n_rejected_q = n_total - n_pos
    logger.info(
        "Q_back > 0 filter: %d / %d retained (%d rejected as Q ≤ 0)",
        n_pos, n_total, n_rejected_q,
    )

    if n_pos < 5:
        logger.error(
            "Insufficient positive-Q intervals (%d < 5) for map fitting.",
            n_pos,
        )
        return _empty_result()

    # -- Step 4: Fit ASHP maps using existing utility from src.ashp_model ----
    ashp_params = fit_ashp_maps(
        T_out=bc_pos["T_out_c"].values,
        T_sink=bc_pos["T_sink_c"].values,
        Q_meas_kwh=bc_pos["Q_back_kwh"].values,
        P_meas_kwh=bc_pos["P_meas_kwh"].values,
        dt_h=dt_h,
        apply_high_load_filter=cfg.apply_high_load_filter,
    )

    # -- Step 5: Compute reference-point COPs for diagnostics ----------------
    ref_points = [
        ("A-3/W55", -3.0, 55.0),
        ("A7/W55",   7.0, 55.0),
        ("A-3/W35", -3.0, 35.0),
    ]
    ref_cops = {}
    for label, t_out, t_sink in ref_points:
        cop_val = float(predict_cop(
            np.array([t_out]), np.array([t_sink]), ashp_params,
        )[0])
        ref_cops[f"cop_at_{label}"] = round(cop_val, 3)

    # Back-calculated COP statistics.
    # Arithmetic mean is sensitive to a small number of intervals where P_meas
    # is just above the ASHP-on threshold (e.g. 0.015 kWh in a 30-min slot
    # where the ASHP ran only briefly), producing physically implausible COPs.
    # mean_back_cop is therefore computed after capping at _COP_REPORT_CAP;
    # the cap is documented in the output and n_cop_outliers records how many
    # intervals were excluded from the mean (they remain in the fit data).
    _COP_REPORT_CAP = 8.0
    cop_back = bc_pos["Q_back_kwh"] / bc_pos["P_meas_kwh"].clip(lower=1e-3)
    n_cop_outliers = int((cop_back > _COP_REPORT_CAP).sum())
    mean_cop_capped = round(float(cop_back.clip(upper=_COP_REPORT_CAP).mean()), 3)

    # -- Step 6: Build result dictionary -------------------------------------
    result = {
        "ashp": {
            "a": [round(float(v), 6) for v in ashp_params.a],
            "b": [round(float(v), 6) for v in ashp_params.b],
        },
        "identification": {
            "n_intervals_accepted": n_total,
            "n_intervals_q_positive": n_pos,
            "n_intervals_q_rejected": n_rejected_q,
            "n_windows": len(windows),
            "mean_back_cop": mean_cop_capped,
            "mean_back_cop_cap": _COP_REPORT_CAP,
            "n_cop_outliers": n_cop_outliers,
            "median_back_cop": round(float(cop_back.median()), 3),
            **ref_cops,
            "thresholds": {
                "ashp_off_kwh": cfg.ashp_off_kwh,
                "st_off_kwh": cfg.st_off_kwh,
                "imm_off_kwh": cfg.imm_off_kwh,
                "hx_on_c": cfg.sh_off_c,
                "draw_delta_c": cfg.draw_delta_c,
                "min_ashp_intervals": cfg.min_ashp_intervals,
                "apply_high_load_filter": cfg.apply_high_load_filter,
            },
            "ua_loss_used": [round(float(v), 8) for v in ua_loss],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    # -- Step 7: Save to disk ------------------------------------------------
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "ashp_fit.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info("Saved ASHP fit to %s", out_path)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_result() -> Dict:
    """Return an empty result dict when fitting cannot proceed."""
    return {
        "ashp": {"a": [0.0] * 4, "b": [0.0] * 4},
        "identification": {"error": "insufficient_data"},
    }
