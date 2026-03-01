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

logger = logging.getLogger(__name__)


@dataclass
class IdentificationResult:
    """Container for fitted parameters and diagnostics."""
    tank_params: tank_model.TankParams
    ashp_params: ashp_model.ASHPParams
    hx_effectiveness: float
    cost_history: list[float]


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
    Q_ashp_kw = ashp_model.predict_capacity(df["t_amb_c"].values, T_sink, ashp_p)
    # But only deliver heat when ASHP is running (inst > threshold)
    ashp_running = df["ashp_inst_kwh"].fillna(0).values > 0.01
    Q_ashp = np.where(ashp_running, Q_ashp_kw * dt_h, 0.0)

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
    """Fit tank parameters against measured node temperatures.

    Uses ``least_squares`` with soft-L1 loss for robustness.
    """
    T_meas = inputs["T_meas"]
    Q_st   = inputs["Q_st"]
    Q_ashp = inputs["Q_ashp"]
    Q_imm  = inputs["Q_imm"]
    T_amb  = inputs["T_amb"]
    # We have N measurement rows; simulate N-1 steps from T_meas[0]
    # using inputs[0:N-1] to predict T_meas[1:N].
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
        T0 = T_meas[0]
        T_sim = tank_model.simulate(
            T0, Q_st[:steps], Q_ashp[:steps], Q_imm[:steps], T_amb[:steps], p,
        )
        # T_sim is (steps+1, 4); compare T_sim[1:] with T_meas[1:N]
        err = (T_sim[1:] - T_meas[1:N]).ravel()

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

    # Step 1: Fit ASHP maps on training data
    T_sink_train = ashp_model.sink_proxy(
        df_train["tank_mid_c"].values,
        df_train["tank_top_c"].values,
    )
    ashp_p = ashp_model.fit_ashp_maps(
        T_amb=df_train["t_amb_c"].values,
        T_sink=T_sink_train,
        Q_meas_kwh=None,   # no direct condenser measurement
        P_meas_kwh=df_train["ashp_inst_kwh"].values,
    )

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
