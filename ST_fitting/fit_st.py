"""
ST_fitting.fit_st – Fit Q_sol energy and T_flow regression maps from
ST-only windows and historical GTI data.

Replaces the legacy f_st distribution approach.  Two OLS regressions are
fitted using interval-level data extracted from the accepted ST-only
windows detected by ``ST_fitting.detector``:

1. **Q_sol energy map** (trilinear in GTI, T_bottom, T_out):
    Q_meas_kwh = q0 + q1 × GTI + q2 × T_bottom + q3 × T_out
                          [kWh per interval]
   Used at deployment to predict ST heat delivery from a GTI forecast.

2. **T_flow bilinear map** (no interaction term):
       T_flow = b0 + b1 × T_bottom + b2 × GTI     [°C]
   Used to compute the dynamic per-node heat allocation weights inside
   the tank model.

GTI is loaded from an external CSV (``hist_GTI_5min.csv``) and joined to
the training slice by **exact timestamp only** (no interpolation/fill).
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

logger = logging.getLogger(__name__)

_RHO = 1000.0  # kg/m^3
_CP = 4.186    # kJ/(kg*K)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_gti(gti_csv_path: str | Path) -> pd.Series:
    """Load hist_GTI_5min.csv and return a Series with DatetimeIndex.

    Expects columns: ``time`` (parseable datetime), ``GTI`` (W/m²).
    Returns a ``pd.Series`` named ``'GTI'`` indexed by UTC-naive Timestamp.
    """
    gti_df = pd.read_csv(
        gti_csv_path,
        parse_dates=["time"],
        index_col="time",
    )
    return gti_df["GTI"]


def _empty_result() -> tuple[Dict, pd.DataFrame]:
    """Return a placeholder result when fitting cannot proceed."""
    return (
        {"identification": {"error": "insufficient_data"}},
        pd.DataFrame(),
    )


def _r2_from_fit(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return in-sample R^2 for a fitted model."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _fit_q_linear(
    q_data: pd.DataFrame,
    *,
    q_col: str,
) -> tuple[np.ndarray, float]:
    """Fit Q ~ 1 + GTI + T_bottom + T_out and return (coeffs, r2)."""
    X_q = np.column_stack([
        np.ones(len(q_data)),
        q_data["GTI"].values,
        q_data["tank_bottom_c"].values,
        q_data["t_out_c"].values,
    ])
    y_q = q_data[q_col].values
    coeffs_q, _, _, _ = np.linalg.lstsq(X_q, y_q, rcond=None)
    y_q_pred = X_q @ coeffs_q
    return coeffs_q, _r2_from_fit(y_q, y_q_pred)


def _derive_q_from_flow(reg_df: pd.DataFrame) -> pd.Series:
    """Compute interval ST energy [kWh] from flow and (T_flow - T_return)."""
    flow_l = reg_df["st_flow_l"].fillna(0.0)
    t_flow = reg_df["st_flow_temp_c"].fillna(0.0)
    t_ret = reg_df["st_return_temp_c"].fillna(np.nan)
    delta_t = (t_flow - t_ret).clip(lower=0.0)
    mass_kg = flow_l * (_RHO / 1000.0)
    q_kj = mass_kg * _CP * delta_t
    return (q_kj / 3600.0).clip(lower=0.0)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def fit_q_sol_regression(
    df: pd.DataFrame,
    windows: List[STWindow],
    gti_csv_path: str | Path,
    cfg: Optional[STFitConfig] = None,
    *,
    q_source: str = "power",
    benchmark_sources: bool = True,
    output_dir: Optional[Path] = None,
) -> tuple[Dict, pd.DataFrame]:
    """Fit Q_sol and T_flow regressions from accepted ST-only windows.

    Parameters
    ----------
    df           : Full cleaned DataFrame (DatetimeIndex, 5-min cadence).
                   Must contain ``st_power_kw`` and ``st_flow_temp_c``.
    windows      : List of STWindow objects from ``detect_st_windows()``.
    gti_csv_path : Path to hist_GTI_5min.csv.
    cfg          : STFitConfig (defaults to ``STFitConfig()`` if *None*).
    output_dir   : Directory for ``st_fit.json`` output.
    q_source     : Target source for Q regression, ``"power"`` or ``"derived"``.
    benchmark_sources : If True, compute side-by-side Q fit metrics for both
                        target sources in ``result['identification']['q_source_benchmark']``.

    Returns
    -------
    result : dict
        Matches the ``st_fit.json`` schema.
    reg_df : pd.DataFrame
        Per-interval regression diagnostics (also written to CSV).
    """
    if cfg is None:
        cfg = STFitConfig()

    if q_source not in {"power", "derived"}:
        raise ValueError("q_source must be one of {'power', 'derived'}")

    if not windows:
        logger.error("No ST-only windows available for regression fitting.")
        return _empty_result()

    dt_h = cfg.sampling_minutes / 60.0

    # -- 1. Load GTI and attach to training slice ----------------------------
    gti_series = _load_gti(gti_csv_path)
    n_train = int(len(df) * cfg.train_frac)
    df_train = df.iloc[:n_train].copy()
    df_train["GTI"] = gti_series.reindex(df_train.index)

    # -- 2. Collect interval-level data from accepted windows ----------------
    rows: list[dict] = []
    for w in windows:
        for k in w.indices:
            rows.append({
                "time": df_train.index[k],
                "window_id": w.window_id,
                "GTI": df_train["GTI"].iloc[k],
                "tank_bottom_c": df_train[cfg.node_cols[0]].iloc[k],
                "t_out_c": df_train[cfg.t_out_col].iloc[k],
                "st_flow_l": df_train[cfg.st_flow_col].iloc[k],
                "st_power_kw": df_train[cfg.st_power_col].iloc[k],
                "st_flow_temp_c": df_train[cfg.st_flow_temp_col].iloc[k],
                "st_return_temp_c": df_train[cfg.st_return_temp_col].iloc[k],
            })

    reg_df = pd.DataFrame(rows).set_index("time")
    n_intervals_total = len(reg_df)

    # Compute interval energy targets [kWh]
    reg_df["q_power_kwh"] = reg_df["st_power_kw"].fillna(0.0).clip(lower=0.0) * dt_h
    reg_df["q_derived_kwh"] = _derive_q_from_flow(reg_df)
    q_target_col = "q_power_kwh" if q_source == "power" else "q_derived_kwh"
    reg_df["q_meas_kwh"] = reg_df[q_target_col]
    n_return_temp_available = int(reg_df["st_return_temp_c"].notna().sum())
    n_q_derived_positive = int((reg_df["q_derived_kwh"] > 0).sum())

    # Count rows with no GTI match
    no_gti_mask = reg_df["GTI"].isna()
    n_dropped_no_gti = int(no_gti_mask.sum())
    if n_dropped_no_gti > 0:
        logger.warning(
            "%d of %d intervals have no GTI match and will be excluded.",
            n_dropped_no_gti, n_intervals_total,
        )

    # -- 3. Q_sol regression: Q_meas_kwh ~ GTI + T_bottom + T_out ----------
    q_mask = (
        (~reg_df["GTI"].isna())
        & (reg_df["GTI"] > 0)
        & (~reg_df["q_meas_kwh"].isna())
        & (~reg_df["tank_bottom_c"].isna())
        & (~reg_df["t_out_c"].isna())
    )
    q_data = reg_df.loc[q_mask]
    n_qsol = len(q_data)

    if n_qsol < 5:
        logger.error("Only %d valid points for Q_sol regression (need >= 5).", n_qsol)
        return _empty_result()

    if n_qsol < 20:
        logger.warning("Q_sol regression has only %d points (< 20).", n_qsol)

    coeffs_q, r2_q = _fit_q_linear(q_data, q_col="q_meas_kwh")
    q0, q1, q2, q3 = [float(v) for v in coeffs_q]
    gti_min_val = float(q_data["GTI"].min())
    gti_max_val = float(q_data["GTI"].max())

    logger.info(
        "Q_sol regression: q0=%.6f kWh  q1=%.6f kWh/(W/m²)  "
        "q2=%.6f kWh/°C  q3=%.6f kWh/°C  R²=%.4f  n=%d",
        q0, q1, q2, q3, r2_q, n_qsol,
    )

    q_source_benchmark: Optional[Dict] = None
    if benchmark_sources:
        coeffs_power, r2_power = _fit_q_linear(q_data, q_col="q_power_kwh")
        coeffs_derived, r2_derived = _fit_q_linear(q_data, q_col="q_derived_kwh")
        q_source_benchmark = {
            "power": {
                "r2": round(float(r2_power), 6),
                "q0_kwh": round(float(coeffs_power[0]), 6),
                "q1_kwh_per_wm2": round(float(coeffs_power[1]), 6),
                "q2_kwh_per_c": round(float(coeffs_power[2]), 6),
                "q3_kwh_per_c": round(float(coeffs_power[3]), 6),
            },
            "derived": {
                "r2": round(float(r2_derived), 6),
                "q0_kwh": round(float(coeffs_derived[0]), 6),
                "q1_kwh_per_wm2": round(float(coeffs_derived[1]), 6),
                "q2_kwh_per_c": round(float(coeffs_derived[2]), 6),
                "q3_kwh_per_c": round(float(coeffs_derived[3]), 6),
            },
            "selected_source": q_source,
        }
        logger.info(
            "Q source benchmark: power R²=%.4f, derived R²=%.4f (selected=%s)",
            r2_power,
            r2_derived,
            q_source,
        )

    # Window-level diagnostic (aggregate interval noise)
    q_window = q_data.groupby("window_id", as_index=False).agg(
        q_meas_kwh=("q_meas_kwh", "sum"),
        GTI=("GTI", "mean"),
        tank_bottom_c=("tank_bottom_c", "mean"),
        t_out_c=("t_out_c", "mean"),
    )
    n_qsol_windows = int(len(q_window))
    if n_qsol_windows >= 5:
        X_w = np.column_stack([
            np.ones(n_qsol_windows),
            q_window["GTI"].values,
            q_window["tank_bottom_c"].values,
            q_window["t_out_c"].values,
        ])
        y_w = q_window["q_meas_kwh"].values
        y_w_pred = X_w @ coeffs_q
        r2_q_window = _r2_from_fit(y_w, y_w_pred)
        logger.info(
            "Q_sol window-level diagnostic: R²=%.4f  n_windows=%d",
            r2_q_window, n_qsol_windows,
        )
    else:
        r2_q_window = float("nan")
        logger.warning(
            "Q_sol window-level diagnostic skipped: only %d windows (<5).",
            n_qsol_windows,
        )

    # -- 4. T_flow regression: T_flow ~ T_bottom + GTI (bilinear, no cross) -
    t_mask = (
        (~reg_df["GTI"].isna())
        & (reg_df["GTI"] > 0)
        & (~reg_df["st_flow_temp_c"].isna())
        & (~reg_df["tank_bottom_c"].isna())
    )
    t_data = reg_df.loc[t_mask]
    n_tflow = len(t_data)

    if n_tflow < 5:
        logger.error("Only %d valid points for T_flow regression (need >= 5).", n_tflow)
        return _empty_result()

    # Design matrix [1, T_bottom, GTI]
    X = np.column_stack([
        np.ones(n_tflow),
        t_data["tank_bottom_c"].values,
        t_data["GTI"].values,
    ])
    y = t_data["st_flow_temp_c"].values
    coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    b0_c, b1, b2 = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

    # R² for T_flow
    y_pred_t = X @ coeffs
    ss_res = float(np.sum((y - y_pred_t) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_t = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    logger.info(
        "T_flow regression: b0=%.4f°C  b1=%.4f  b2=%.6f °C/(W/m²)  R²=%.4f  n=%d",
        b0_c, b1, b2, r2_t, n_tflow,
    )

    # -- 5. Add predicted columns to reg_df ----------------------------------
    reg_df["q_pred_kwh"] = (
        q0
        + q1 * reg_df["GTI"]
        + q2 * reg_df["tank_bottom_c"]
        + q3 * reg_df["t_out_c"]
    ).clip(lower=0.0)
    reg_df["t_flow_pred_c"] = b0_c + b1 * reg_df["tank_bottom_c"] + b2 * reg_df["GTI"]

    # -- 6. Build result dict ------------------------------------------------
    result: Dict = {
        "Q_sol": {
            "q0_kwh": round(q0, 6),
            "q1_kwh_per_wm2": round(q1, 6),
            "q2_kwh_per_c": round(q2, 6),
            "q3_kwh_per_c": round(q3, 6),
            "r2": round(r2_q, 6),
            "n_points": n_qsol,
            "gti_range_wm2": [round(gti_min_val, 1), round(gti_max_val, 1)],
        },
        "T_flow": {
            "b0_c": round(b0_c, 6),
            "b1_per_c": round(b1, 6),
            "b2_c_per_wm2": round(b2, 6),
            "r2": round(r2_t, 6),
            "n_points": n_tflow,
        },
        "activation": {
            "gti_min_wm2": cfg.gti_min_wm2,
            "t_bottom_max_c": cfg.t_bottom_max_c,
        },
        "identification": {
            "n_windows": len(windows),
            "n_intervals_total": n_intervals_total,
            "n_intervals_used_qsol": n_qsol,
            "n_intervals_used_tflow": n_tflow,
            "n_windows_used_qsol": n_qsol_windows,
            "r2_qsol_window": round(r2_q_window, 6) if np.isfinite(r2_q_window) else None,
            "q_source": q_source,
            "q_target_column": q_target_col,
            "n_return_temp_available": n_return_temp_available,
            "n_q_derived_positive": n_q_derived_positive,
            "n_intervals_dropped_no_gti": n_dropped_no_gti,
            "train_frac": cfg.train_frac,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    if q_source_benchmark is not None:
        result["identification"]["q_source_benchmark"] = q_source_benchmark

    # -- 7. Save JSON --------------------------------------------------------
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_json = output_dir / "st_fit.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info("Saved ST fit to %s", out_json)

    return result, reg_df
