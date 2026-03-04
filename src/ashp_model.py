"""
ASHP performance maps (control-oriented).

A performance map is a compact mathematical equation that describes how the heat pump
behaves without simulating its internal physics.

Outputs: Q_cond (condenser heat output) and P_elec (electrical power input).
Inputs: T_out (outdoor air temperature) and T_sink (tank sink-proxy temperature).  The sink-proxy is a weighted average
of the mid and top node temperatures, representing the effective sink temperature seen by the ASHP.

Compact parametric relationships for capacity and electrical power
as functions of ambient temperature and tank sink-proxy temperature.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)

# This percentile is used later for filtering
# This ensures the map is fitted to seady full-load operation, not partial or start-up which would distort results.
HIGH_LOAD_PERCENTILE = 75


@dataclass
class ASHPParams:
    """Identified ASHP map parameters.
    
    Assume a bilinear form for both capacity and power maps:

    Map input temperature is outdoor air temperature (T_out).

    Capacity  Q̇_cond = a0 + a1·T_out + a2·T_sink + a3·T_out·T_sink
    Power     P_elec = b0 + b1·T_out + b2·T_sink + b3·T_out·T_sink
    """
    a: np.ndarray = field(default_factory=lambda: np.array([8.0, 0.1, -0.05, 0.0]))
    b: np.ndarray = field(default_factory=lambda: np.array([3.0, -0.02, 0.03, 0.0]))


def sink_proxy(
    T_mid: np.ndarray,
    T_top: np.ndarray,
    w_mid: float = 0.5,
    w_top: float = 0.5,
) -> np.ndarray:
    """Weighted average of mid and top node temperatures."""
    return w_mid * np.asarray(T_mid) + w_top * np.asarray(T_top)


def predict_capacity(T_out: np.ndarray, T_sink: np.ndarray, p: ASHPParams) -> np.ndarray:
    """Predict condenser heat output [kW]."""
    a = p.a
    T_a, T_s = np.asarray(T_out, dtype=float), np.asarray(T_sink, dtype=float)
    return np.maximum(a[0] + a[1] * T_a + a[2] * T_s + a[3] * T_a * T_s, 0.0)


def predict_power(T_out: np.ndarray, T_sink: np.ndarray, p: ASHPParams) -> np.ndarray:
    """Predict electrical power [kW]."""
    b = p.b
    T_a, T_s = np.asarray(T_out, dtype=float), np.asarray(T_sink, dtype=float)
    return np.maximum(b[0] + b[1] * T_a + b[2] * T_s + b[3] * T_a * T_s, 0.1)


def predict_cop(T_out: np.ndarray, T_sink: np.ndarray, p: ASHPParams) -> np.ndarray:
    """Return COP = Q̇_cond / P_elec."""
    q = predict_capacity(T_out, T_sink, p)
    pel = predict_power(T_out, T_sink, p)
    return q / pel


def fit_ashp_maps(
    T_out: np.ndarray,
    T_sink: np.ndarray,
    Q_meas_kwh: np.ndarray,
    P_meas_kwh: np.ndarray,
    dt_h: float = 0.5,
) -> ASHPParams:
    """Fit ASHP capacity and power maps from measured interval energies.
    Stage 1: Data Filtering
    Stage 2: Power Map Fitting
    Stage 3: Capacity Map Fitting

    Parameters
    ----------
    T_out, T_sink : outdoor air temperature and sink-proxy arrays [°C].
    Q_meas_kwh : measured condenser heat per interval [kWh] (may be unknown;
                 pass NaN to skip capacity fitting – power only).
    P_meas_kwh : measured ASHP electrical energy per interval [kWh].
    dt_h : interval length in hours (default 0.5).

    Returns
    -------
    ASHPParams with fitted coefficients.
    """
    T_a = np.asarray(T_out, dtype=float)
    T_s = np.asarray(T_sink, dtype=float)
    P_meas = np.asarray(P_meas_kwh, dtype=float)

    # Mask: only intervals where ASHP was running at substantial load.
    # Use a high percentile threshold so we fit steady-state power
    # (not partial duty-cycle intervals).
    valid = np.isfinite(P_meas) & (P_meas > 0.05) & np.isfinite(T_a) & np.isfinite(T_s) # Removes intervals with NaN Temps or near-zero ASHP power
    # If there are enough intervals (>50) we can apply high-load filter (>75%)
    if valid.sum() > 50:
        p75 = np.percentile(P_meas[valid], HIGH_LOAD_PERCENTILE)
        mask = valid & (P_meas >= p75)
    else:
        mask = valid
    T_a_f, T_s_f, P_f = T_a[mask], T_s[mask], P_meas[mask]

    # Convert interval kWh → average kW
    P_kw = P_f / dt_h

    # Fit power map: P_elec = b0 + b1*T_a + b2*T_s + b3*T_a*T_s
    X = np.column_stack([np.ones(len(T_a_f)), T_a_f, T_s_f, T_a_f * T_s_f])

    # Use Ordinary Least Squares (OLS) for a good initial guess, then refine with robust loss
    # OLS finds the b that minimises the sum of squared errors between X @ b (@ is matrix multiplication).
    b_ols, _, _, _ = np.linalg.lstsq(X, P_kw, rcond=None)
    b_lo = np.array([-20.0, -0.5, -0.5, -0.02])
    b_hi = np.array([20.0,   0.5,  0.5,  0.02])
    b_init = np.clip(b_ols, b_lo + 1e-6, b_hi - 1e-6) # ensure initial guess is within bounds for least_squares

    def power_residuals(b):
        pred = X @ b
        pred = np.maximum(pred, 0.1) 
        return pred - P_kw

    # 
    res_b = least_squares(
        power_residuals, b_init,
        bounds=(b_lo, b_hi), # constrain to physically plausible values
        loss="soft_l1", # less sensitive to outliers than squared loss
    )

    params = ASHPParams()
    params.b = res_b.x
    logger.info("ASHP power map coefficients: %s", params.b)

    # Capacity fit: if Q_meas available
    Q_meas = np.asarray(Q_meas_kwh, dtype=float) if Q_meas_kwh is not None else np.full_like(P_meas, np.nan)
    mask_q = mask & np.isfinite(Q_meas) & (Q_meas > 0.01)

    # If direct heat output measurements (Q_meas_kwh) are available and there are enough valid points (>20),
    # the a coefficients are fitted directly from data — the preferred approach.
    if mask_q.sum() > 20:
        T_a_q, T_s_q = T_a[mask_q], T_s[mask_q]
        Q_kw = Q_meas[mask_q] / dt_h
        Xq = np.column_stack([np.ones(len(T_a_q)), T_a_q, T_s_q, T_a_q * T_s_q])
        a_init = np.array([8.0, 0.1, -0.05, 0.0])

        def cap_residuals(a):
            pred = Xq @ a
            pred = np.maximum(pred, 0.0)
            return pred - Q_kw

        res_a = least_squares(cap_residuals, a_init, loss="soft_l1")
        params.a = res_a.x
        logger.info("ASHP capacity map coefficients: %s", params.a)
    # Otherwise they're estimated from an assumed average COP
    else:
        # Estimate capacity from power × assumed COP
        avg_cop = 3.0
        params.a = params.b * avg_cop
        logger.info("ASHP capacity estimated from power × COP=%.1f", avg_cop)

    return params
