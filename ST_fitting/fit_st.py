"""
ST_fitting.fit_st – Regression-based ST energy and flow-temperature fitting.

Models fitted
-------------
Q_sol_kwh  = q0 + q1 * GTI                    (interval ST energy [kWh])
T_flow     = b0 + b1 * T_bottom + b2 * GTI    (bilinear flow-temperature [°C])

Both regressions are fitted on per-interval data extracted from clean
ST-only windows detected by the detector module.  GTI (Global Tilted
Irradiance) is loaded from a separate CSV and joined on timestamp.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ST_fitting.config import STFitConfig
from ST_fitting.detector import STWindow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: load GTI
# ---------------------------------------------------------------------------

def _load_gti(gti_csv_path: Path) -> pd.Series:
    """Load the GTI time series and return a Series indexed by datetime.

    Expected CSV columns: ``time``, ``GTI`` (W/m²).
    """
    gti_df = pd.read_csv(gti_csv_path, parse_dates=["time"])
    gti_df = gti_df.set_index("time").sort_index()
    if "GTI" not in gti_df.columns:
        raise KeyError("GTI column not found in %s" % gti_csv_path)
    return gti_df["GTI"]


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def fit_q_sol_regression(
    df: pd.DataFrame,
    windows: List[STWindow],
    gti_csv_path: Path,
    cfg: Optional[STFitConfig] = None,
    *,
    output_dir: Optional[Path] = None,
) -> Tuple[Dict, pd.DataFrame]:
    """Fit Q_sol and T_flow regressions from ST-only window intervals.

    Parameters
    ----------
    df : pd.DataFrame
        Full cleaned DataFrame.
    windows : list[STWindow]
        Accepted ST-only windows from the detector.
    gti_csv_path : Path
        Path to CSV with ``time`` and ``GTI`` columns.
    cfg : STFitConfig, optional
    output_dir : Path, optional
        Directory for ``st_fit.json``.

    Returns
    -------
    result : dict
        Nested dict with Q_sol, T_flow coefficients, R² and metadata.
    reg_df : pd.DataFrame
        Per-interval regression data for diagnostics.
    """
    if cfg is None:
        cfg = STFitConfig()

    dt_h = cfg.sampling_minutes / 60.0

    # -- 1. Load GTI and join ------------------------------------------------
    gti_series = _load_gti(gti_csv_path)

    # -- 2. Training slice ---------------------------------------------------
    n_train = int(len(df) * cfg.train_frac)
    df_train = df.iloc[:n_train]

    # -- 3. Collect per-interval data from windows ---------------------------
    rows = []
    for w in windows:
        for k in w.indices:
            ts = df_train.index[k]
            if ts not in gti_series.index:
                continue
            gti_val = float(gti_series.loc[ts])
            st_power = float(df_train[cfg.st_power_col].iloc[k])
            q_meas = st_power * dt_h  # kWh
            t_bottom = float(df_train[cfg.node_cols[0]].iloc[k])
            t_flow = float(df_train[cfg.st_flow_temp_col].iloc[k])
            rows.append({
                "time": ts,
                "GTI": gti_val,
                "q_meas_kwh": q_meas,
                "t_bottom_c": t_bottom,
                "st_flow_temp_c": t_flow,
            })

    reg_df = pd.DataFrame(rows)
    n_dropped = sum(len(w.indices) for w in windows) - len(reg_df)
    logger.info(
        "Regression dataset: %d intervals (%d dropped for missing GTI).",
        len(reg_df), n_dropped,
    )

    if len(reg_df) < 3:
        logger.error("Too few intervals (%d) for regression.", len(reg_df))
        return _empty_result()

    # -- 4. Q_sol regression: q_meas_kwh ~ GTI ------------------------------
    slope_q, intercept_q, r_q, p_q, se_q = stats.linregress(
        reg_df["GTI"].values, reg_df["q_meas_kwh"].values,
    )
    logger.info(
        "Q_sol regression: q0=%.6f  q1=%.6e  R²=%.4f  n=%d",
        intercept_q, slope_q, r_q ** 2, len(reg_df),
    )

    # -- 5. T_flow regression: st_flow_temp ~ [1, T_bottom, GTI] ------------
    X = np.column_stack([
        np.ones(len(reg_df)),
        reg_df["t_bottom_c"].values,
        reg_df["GTI"].values,
    ])
    y_flow = reg_df["st_flow_temp_c"].values
    coeffs, residuals, rank, sv = np.linalg.lstsq(X, y_flow, rcond=None)
    b0_c, b1, b2_c = coeffs

    # R² for T_flow
    y_pred_flow = X @ coeffs
    ss_res = float(np.sum((y_flow - y_pred_flow) ** 2))
    ss_tot = float(np.sum((y_flow - y_flow.mean()) ** 2))
    r2_flow = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    logger.info(
        "T_flow regression: b0=%.4f  b1=%.4f  b2=%.6f  R²=%.4f  n=%d",
        b0_c, b1, b2_c, r2_flow, len(reg_df),
    )

    # -- 6. Build result dict ------------------------------------------------
    result = {
        "Q_sol": {
            "q0_kwh": round(float(intercept_q), 6),
            "q1_kwh_per_wm2": round(float(slope_q), 8),
            "r2": round(float(r_q ** 2), 6),
            "p_value": round(float(p_q), 6),
            "se_slope": round(float(se_q), 8),
            "n_intervals": int(len(reg_df)),
        },
        "T_flow": {
            "b0_c": round(float(b0_c), 6),
            "b1": round(float(b1), 6),
            "b2_c_per_wm2": round(float(b2_c), 8),
            "r2": round(float(r2_flow), 6),
            "n_intervals": int(len(reg_df)),
        },
        "activation": {
            "gti_min_wm2": cfg.gti_min_wm2,
            "t_bottom_max_c": cfg.t_bottom_max_c,
        },
        "identification": {
            "n_windows": len(windows),
            "n_intervals_used": int(len(reg_df)),
            "n_intervals_dropped": int(n_dropped),
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

    # -- 7. Save outputs -----------------------------------------------------
    if output_dir is not None:
        import json
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_json = output_dir / "st_fit.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info("Saved ST fit to %s", out_json)

    return result, reg_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_result() -> Tuple[Dict, pd.DataFrame]:
    return (
        {"identification": {"error": "insufficient_data"}},
        pd.DataFrame(),
    )
