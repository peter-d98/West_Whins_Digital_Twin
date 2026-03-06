"""
Parameter identification for the Stage-1 digital twin.

Two-step procedure:
  1. Fit ASHP maps on intervals with no immersion and low ST.
  2. Fit tank parameters (with ASHP heat derived from the map).

Joint refinement with regularisation is also available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from . import ashp_model, solar_thermal, tank_model
from .tank_model import NODE_CAP, TankParams

logger = logging.getLogger(__name__)


@dataclass
class IdentificationResult:
    """Container for fitted parameters and diagnostics."""
    tank_params: tank_model.TankParams
    ashp_params: ashp_model.ASHPParams
    hx_effectiveness: float
    cost_history: list[float]


def back_calculate_ashp_heat(
    df: pd.DataFrame,
    st_col: str = "st_kwh",
    dt_s: float = 1800.0,
) -> pd.Series:
    """Back-calculate ASHP heat delivery [kWh] for ASHP-only intervals.

    Returns a Series of length len(df) with NaN for non-ASHP-only intervals.

    An interval is considered ASHP-only if:
      - ashp_inst_kwh > 0.05  (ASHP was running)
      - imm_tot_inst_kwh < 0.01  (immersion heater was off)
      - st_kwh < 0.05  (negligible solar thermal)
      - all four tank temperature columns are finite at both this
        row and the previous row

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with tank temperatures, ASHP, immersion, and ST data.
    st_col : str
        Name of the solar-thermal energy column.
    dt_s : float
        Interval length in seconds (default 1800).

    Returns
    -------
    pd.Series
        Back-calculated ASHP heat [kWh]; NaN for non-ASHP-only intervals.
    """
    tank_cols = ["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]

    ashp_on = df["ashp_inst_kwh"].fillna(0) > 0.05
    imm_off = df["imm_tot_inst_kwh"].fillna(0) < 0.01
    st_low = df[st_col].fillna(0) < 0.05 if st_col in df.columns else pd.Series(True, index=df.index)

    # All four tank temps must be finite at this row and the previous row
    T = df[tank_cols].values
    finite_now = np.all(np.isfinite(T), axis=1)
    finite_prev = np.roll(finite_now, 1)
    finite_prev[0] = False  # first row has no predecessor

    mask = ashp_on & imm_off & st_low & pd.Series(finite_now & finite_prev, index=df.index)

    n_ashp_only = mask.sum()
    logger.info("ASHP-only intervals found: %d", n_ashp_only)

    Q_back = pd.Series(np.nan, index=df.index)

    if n_ashp_only < 50:
        logger.warning(
            "Insufficient ASHP-only intervals (%d < 50) for back-calculation; "
            "fallback to a = b * 3.0 retained.",
            n_ashp_only,
        )
        return Q_back

    # Default UA_loss for standing-loss correction
    ua_loss_default = TankParams().UA_loss

    T_amb = df["t_amb_c"].fillna(df["t_amb_c"].median()).values

    idx = np.where(mask.values)[0]
    for k in idx:
        dT_sum = 0.0
        loss_sum = 0.0
        T_avg = 0.0
        for i in range(4):
            dT_sum += T[k, i] - T[k - 1, i]
            T_avg += T[k - 1, i]
            loss_sum += ua_loss_default[i] * (T[k - 1, i] - T_amb[k]) * dt_s
        T_avg /= 4.0

        Q_kJ = NODE_CAP * dT_sum + loss_sum
        Q_back.iloc[k] = max(Q_kJ / 3600.0, 0.0)

    return Q_back


def prepare_inputs(df: pd.DataFrame, ashp_p: ashp_model.ASHPParams, dt_h: float = 0.5) -> dict:
    """Build arrays needed for tank simulation from a cleaned DataFrame.

    Returns dict with keys: T_meas (N,4), Q_st, Q_ashp, Q_imm, T_amb (all N,).
    """
    T_meas = df[["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]].values

    # ST energy
    if "st_kwh" in df.columns:
        Q_st = df["st_kwh"].fillna(0).values
    else:
        Q_st = solar_thermal.compute_st_energy(df, dt_minutes=dt_h * 60).values

    # ASHP heat from map
    T_sink = ashp_model.sink_proxy(df["tank_mid_c"].values, df["tank_top_c"].values)
    # Use measured electrical × COP from map to derive heat delivered
    cop = ashp_model.predict_cop(df["t_out_c"].values, T_sink, ashp_p)
    P_meas = df["ashp_inst_kwh"].fillna(0).values
    Q_ashp = P_meas * cop  # kWh heat = kWh elec × COP

    # Immersion
    Q_imm = df["imm_tot_inst_kwh"].fillna(0).values

    T_amb = df["t_amb_c"].fillna(df["t_amb_c"].median()).values

    return dict(
        T_meas=T_meas,
        Q_st=Q_st,
        Q_ashp=Q_ashp,
        Q_imm=Q_imm,
        T_amb=T_amb,
    )


def fit_tank_params(
    inputs: dict,
    *,
    max_nfev: int = 300,
    reg_weight: float = 0.01,
) -> tank_model.TankParams:
    """Fit tank parameters using one-step-ahead (teacher-forced) residuals.

    Each step resets to the measured state, so the residuals are the
    one-step prediction errors.  This avoids error accumulation and gives
    stable parameter estimates.
    """
    T_meas = inputs["T_meas"]
    Q_st   = inputs["Q_st"]
    Q_ashp = inputs["Q_ashp"]
    Q_imm  = inputs["Q_imm"]
    T_amb  = inputs["T_amb"]
    N = len(Q_st)
    steps = N - 1

    p0 = tank_model.TankParams()
    x0 = p0.to_vector()
    lb = tank_model.TankParams.lower_bounds()
    ub = tank_model.TankParams.upper_bounds()

    # Clamp x0 within bounds
    x0 = np.clip(x0, lb + 1e-8, ub - 1e-8)

    def residuals(x):
        p = tank_model.TankParams.from_vector(x)
        # One-step-ahead: predict T[k+1] from measured T[k]
        T_pred = np.zeros((steps, 4))
        for k in range(steps):
            T_pred[k] = tank_model.tank_step(
                T_meas[k],
                float(Q_st[k]), float(Q_ashp[k]),
                float(Q_imm[k]), float(T_amb[k]), p,
            )
        err = (T_pred - T_meas[1: steps + 1]).ravel()
        # Regularisation toward defaults
        reg = reg_weight * (x - p0.to_vector())
        return np.concatenate([err, reg])

    result = least_squares(
        residuals, x0,
        bounds=(lb, ub),
        loss="soft_l1",
        f_scale=2.0,
        max_nfev=max_nfev,
        verbose=0,
    )
    logger.info("Tank fit cost: %.2f, nfev: %d", result.cost, result.nfev)
    return tank_model.TankParams.from_vector(result.x)


def run_identification(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.7,
    max_nfev: int = 300,
) -> tuple[IdentificationResult, pd.DataFrame, pd.DataFrame]:
    """Full identification pipeline.

    Returns
    -------
    result : IdentificationResult
    df_train : training slice
    df_val : validation slice
    """
    # Compute ST energy column
    df = df.copy()
    df["st_kwh"] = solar_thermal.compute_st_energy(df)

    # Train/val split by time
    split_idx = int(len(df) * train_frac)
    df_train = df.iloc[:split_idx].copy()
    df_val   = df.iloc[split_idx:].copy()

    logger.info("Train: %d rows, Val: %d rows", len(df_train), len(df_val))

    # Step 1 (Pass 1): Fit ASHP power (b) coefficients only
    T_sink_train = ashp_model.sink_proxy(
        df_train["tank_mid_c"].values,
        df_train["tank_top_c"].values,
    )
    ashp_p = ashp_model.fit_ashp_maps(
        T_out=df_train["t_out_c"].values,
        T_sink=T_sink_train,
        Q_meas_kwh=None,
        P_meas_kwh=df_train["ashp_inst_kwh"].values,
    )

    # Step 1 (Pass 2): Back-calculate heat delivery and re-fit both a and b
    Q_back = back_calculate_ashp_heat(df_train)
    if Q_back.notna().sum() >= 50:
        ashp_p = ashp_model.fit_ashp_maps(
            T_out=df_train["t_out_c"].values,
            T_sink=T_sink_train,
            Q_meas_kwh=Q_back.values,
            P_meas_kwh=df_train["ashp_inst_kwh"].values,
        )
        logger.info("ASHP capacity fitted from back-calculated heat data.")
    else:
        logger.warning("Insufficient ASHP-only intervals for back-calculation; "
                       "fallback to a = b * 3.0 retained.")

    # Step 2: Fit tank on training data
    train_inputs = prepare_inputs(df_train, ashp_p)
    tank_p = fit_tank_params(train_inputs, max_nfev=max_nfev)

    result = IdentificationResult(
        tank_params=tank_p,
        ashp_params=ashp_p,
        hx_effectiveness=1.0,
        cost_history=[],
    )
    return result, df_train, df_val
