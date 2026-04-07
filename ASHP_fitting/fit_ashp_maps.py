"""
ASHP_fitting.fit_ashp_maps – Back-calculate ASHP heat and fit performance maps.

This module contains:
  1. ``back_calculate_q_ashp`` — computes condenser heat delivery from the
     4-node energy balance using measured tank temperatures and UA_loss priors
     from ``ua_fit.json``.  Returns one row *per window* (aggregated), not one
     row per interval.
  2. ``fit_ashp`` — the main entry point that back-calculates Q, filters
     windows with net Q > 0, and delegates to ``src.ashp_model.fit_ashp_maps``
     for the bilinear map fitting.

Back-calculation physics
------------------------
For each accepted ASHP-only interval *k* within a window:

    Q_k_kJ = Σ_i NODE_CAP_i × (T_i[k] − T_i[k−1])
              + Σ_i UA_loss_i × (T_i[k−1] − T_amb[k]) × dt_s

    Q_k_kWh = Q_k_kJ / 3600

The window-level heat and power are then summed:

    Q_window = Σ_k Q_k_kWh
    P_window = Σ_k P_meas_k

Fitting uses (Q_window, P_window, mean T_out, mean T_sink) per window.

Aggregating over the full window is robust to internal stratification
redistribution.  When the ASHP circulation pump first starts it displaces
hot top-node water into the mid node (a "collapse" step).  The collapse step
has a large negative ΔT in the top node and a large positive ΔT in the mid
node; when both are included in the window sum they cancel correctly, so the
net Q reflects only genuine ASHP heat input rather than internal rearrangement.

The ``mid_rising`` detector gate (``tank_mid_c.diff() > mid_rising_c``) is
used instead of the old ``hx_on`` (``d_top > sh_off_c``) gate precisely
because it admits the collapse step (mid jumps ~5 °C) while still rejecting
space-heating operation (mid flat or falling) and pump-down steps (mid
falling).

NODE_CAP is per-node here (not the uniform value from tank_model) because
the ASHP HX draws from the mid node and returns to the top node — the bottom
node is not directly charged.  The tank geometry used here is:
  - Bottom node : 170 L  (NODE_CAP = 711.62 kJ/K)
  - Mid, Mid-Hi, Top : 380 L split equally → 126.67 L each
                       (NODE_CAP = 530.21 kJ/K each)

Units
-----
- NODE_CAP_ASHP : array [kJ/K], shape (4,), bottom→top
- dt_s          : seconds (300 for 5-min data)
- UA_loss       : kW/K — so UA_loss × ΔT × dt_s gives kJ
- Q_ashp        : kWh per window
- P_meas        : kWh per window (sum of measured ASHP electricity)
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
    *,
    ua_adj: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Back-calculate ASHP condenser heat aggregated per window.

    For each window all accepted intervals are summed so that internal
    stratification-redistribution steps (e.g. the collapse step where hot
    top-node water displaces into the mid node when the circulation pump
    starts) cancel correctly in the energy balance.

    The energy balance is taken over the 3-node sub-system (nodes 1–3: mid,
    mid-hi, top).  Heat conducted from the mid node to the bottom node
    crosses this boundary and is not captured by the temperature rise of
    nodes 1–3.  When ``ua_adj`` is supplied, the boundary flux is added back:

        boundary_kJ_k = ua_adj[0] × (T_mid[k−1] − T_bot[k−1]) × dt_s

    This correction is always positive during charging (T_mid > T_bot).

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
    ua_adj : array of 3 UA_adj values [kW/K] (b-m, m-mh, mh-t), optional.
        When provided, the bottom-boundary conduction correction is applied
        using ``ua_adj[0]``.  Pass ``None`` to skip the correction (legacy
        behaviour).

    Returns
    -------
    pd.DataFrame
        One row per window.  Columns:
        ``window_id``, ``start``, ``end``, ``n_intervals``,
        ``Q_back_kwh``, ``P_meas_kwh``, ``T_out_c``, ``T_sink_c``.
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
        Q_window_kJ = 0.0
        P_window_kwh = 0.0
        t_out_vals = []

        # Pre-window T_sink: the state immediately before the first interval.
        # w.indices[0] is the collapse step where T_mid jumps as the HX pump
        # starts — T[w.indices[0]] already reflects that redistribution.
        # T[w.indices[0] - 1] is the exogenous tank condition the refrigerant
        # cycle was actually set against at startup, and is free of the
        # endogeneity bias that arises from averaging T_sink over the window
        # (where high heat delivery and rising T_sink are positively correlated).
        first_k = w.indices[0]
        t_sink_pre = float(sink_proxy(T[first_k - 1, 1], T[first_k - 1, 3]))

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

            # Boundary correction: heat conducted from mid (node 1) to bottom
            # (node 0) via UA_adj[0] leaves the 3-node accounting boundary and
            # must be added back so Q_window_kJ reflects all ASHP delivery.
            boundary_kJ = 0.0
            if ua_adj is not None:
                boundary_kJ = float(ua_adj[0]) * (T[k - 1, 1] - T[k - 1, 0]) * dt_s

            Q_window_kJ += storage_kJ + loss_sum + boundary_kJ
            P_window_kwh += float(P_ashp[k])
            t_out_vals.append(float(T_out[k]))

        records.append({
            "window_id": w.window_id,
            "start": w.start,
            "end": w.end,
            "n_intervals": w.n_intervals,
            "Q_back_kwh": Q_window_kJ / 3600.0,
            "P_meas_kwh": P_window_kwh,
            "COP_back": (Q_window_kJ / 3600.0) / P_window_kwh if P_window_kwh > 1e-3 else np.nan,
            "T_out_c": float(np.mean(t_out_vals)),
            "T_sink_c": t_sink_pre,
        })

    result_df = pd.DataFrame(records)
    logger.info(
        "Back-calculated Q for %d windows (%d total intervals).",
        len(result_df), sum(w.n_intervals for w in windows),
    )
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
    ua_adj: Optional[np.ndarray] = None,
    output_dir: Optional[Path] = None,
) -> Dict:
    """Back-calculate ASHP heat per window, filter net Q > 0, fit bilinear maps.

    Parameters
    ----------
    df : pd.DataFrame
        Full cleaned DataFrame.
    windows : list[ASHPWindow]
        Accepted ASHP-only windows from the detector.
    ua_loss : array of shape (4,) — UA_loss priors [kW/K].
    cfg : ASHPFitConfig, optional
    ua_adj : array of shape (3,), optional
        Inter-node conductances [kW/K] from ``ua_fit.json``.  When provided,
        the bottom-boundary correction is applied in back-calculation (see
        ``back_calculate_q_ashp``).
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

    # -- Step 2: Back-calculate Q_ashp, aggregated per window ----------------
    if ua_adj is not None:
        logger.info(
            "Applying bottom-boundary correction with UA_adj[0]=%.5f kW/K",
            float(ua_adj[0]),
        )
    bc_df = back_calculate_q_ashp(df_train, windows, ua_loss, cfg, ua_adj=ua_adj)

    if bc_df.empty:
        logger.error("No ASHP-only windows available for back-calculation.")
        return _empty_result()

    # -- Step 3: Filter windows with net Q > 0 -------------------------------
    # A window with net Q ≤ 0 delivered no useful heat to the tank (e.g. a
    # very short run dominated by standby losses); exclude from fitting.
    n_total = len(bc_df)
    bc_pos = bc_df[bc_df["Q_back_kwh"] > 0.0].copy()
    n_pos = len(bc_pos)
    n_rejected_q = n_total - n_pos
    logger.info(
        "Net Q > 0 filter: %d / %d windows retained (%d rejected)",
        n_pos, n_total, n_rejected_q,
    )

    if n_pos < 5:
        logger.error(
            "Insufficient positive-Q windows (%d < 5) for map fitting.",
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

    # -- Step 5: Compute A7/W49 COP and back-calculated COP statistics -------
    cop_a7w49 = round(float(predict_cop(
        np.array([7.0]), np.array([49.0]), ashp_params,
    )[0]), 3)

    # Window-level COP statistics (Q_window / P_window).
    _COP_REPORT_CAP = 8.0
    cop_back = bc_pos["Q_back_kwh"] / bc_pos["P_meas_kwh"].clip(lower=1e-3)
    n_cop_outliers = int((cop_back > _COP_REPORT_CAP).sum())
    cop_back_capped = cop_back.clip(upper=_COP_REPORT_CAP)
    mean_cop_capped = round(float(cop_back_capped.mean()), 3)
    std_cop_capped = round(float(cop_back_capped.std()), 3)

    # -- Step 6: Build result dictionary -------------------------------------
    result = {
        "ashp": {
            "c": [round(float(v), 6) for v in ashp_params.c],
        },
        "identification": {
            "n_windows_total": n_total,
            "n_windows_accepted": n_pos,
            "n_windows_q_rejected": n_rejected_q,
            "n_intervals_total": int(sum(w.n_intervals for w in windows)),
            "cop_at_A7/W49": cop_a7w49,
            "mean_back_cop": mean_cop_capped,
            "std_back_cop": std_cop_capped,
            "mean_back_cop_cap": _COP_REPORT_CAP,
            "n_cop_outliers": n_cop_outliers,
            "median_back_cop": round(float(cop_back.median()), 3),
            "thresholds": {
                "ashp_off_kwh": cfg.ashp_off_kwh,
                "st_off_kwh": cfg.st_off_kwh,
                "imm_off_kwh": cfg.imm_off_kwh,
                "mid_rising_c": cfg.mid_rising_c,
                "draw_delta_c": cfg.draw_delta_c,
                "min_ashp_intervals": cfg.min_ashp_intervals,
                "apply_high_load_filter": cfg.apply_high_load_filter,
            },
            "ua_loss_used": [round(float(v), 8) for v in ua_loss],
            "ua_adj_boundary_used": round(float(ua_adj[0]), 8) if ua_adj is not None else None,
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
        "ashp": {"c": [0.0] * 4},
        "identification": {"error": "insufficient_data"},
    }
