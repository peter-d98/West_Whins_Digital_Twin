"""
ASHP performance maps (control-oriented).

A performance map is a compact mathematical equation that describes how the heat pump
behaves without simulating its internal physics.

Output: COP (coefficient of performance) as a function of operating conditions.
Inputs: T_out (outdoor air temperature) and T_sink (tank sink-proxy temperature).
The sink-proxy is a weighted average of the mid and top node temperatures,
representing the effective sink temperature seen by the ASHP.

For MPC: given a measured or scheduled electrical input P_elec, the heat
delivered is Q = P_elec × COP(T_out, T_sink).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)

# This percentile is used later for filtering
# This ensures the map is fitted to steady full-load operation, not partial or start-up which would distort results.
HIGH_LOAD_PERCENTILE = 75


@dataclass
class ASHPParams:
    """Identified ASHP map parameters.

    Bilinear COP map:

        COP = c0 + c1·T_out + c2·T_sink + c3·T_out·T_sink

    The ``a`` (capacity) and ``b`` (power) arrays are retained for backward
    compatibility but are no longer the primary representation.  New code
    should use ``c`` (COP coefficients) directly.
    """
    # COP map coefficients (primary)
    c: np.ndarray = field(default_factory=lambda: np.array([3.0, 0.05, -0.02, 0.0]))
    # Legacy capacity map coefficients (kept for backward compat)
    a: np.ndarray = field(default_factory=lambda: np.array([8.0, 0.1, -0.05, 0.0]))
    # Legacy power map coefficients (kept for backward compat)
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
    """Predict condenser heat output [kW] (legacy — uses ``a`` coefficients)."""
    a = p.a
    T_a, T_s = np.asarray(T_out, dtype=float), np.asarray(T_sink, dtype=float)
    return np.maximum(a[0] + a[1] * T_a + a[2] * T_s + a[3] * T_a * T_s, 0.0)


def predict_power(T_out: np.ndarray, T_sink: np.ndarray, p: ASHPParams) -> np.ndarray:
    """Predict electrical power [kW] (legacy — uses ``b`` coefficients)."""
    b = p.b
    T_a, T_s = np.asarray(T_out, dtype=float), np.asarray(T_sink, dtype=float)
    return np.maximum(b[0] + b[1] * T_a + b[2] * T_s + b[3] * T_a * T_s, 0.1)


def predict_cop(T_out: np.ndarray, T_sink: np.ndarray, p: ASHPParams) -> np.ndarray:
    """Return COP from the direct COP map.

    Uses the ``c`` coefficients if available (length-4 array); otherwise
    falls back to ``a / b`` ratio for backward compatibility with old JSON
    files that lack a ``c`` key.
    """
    T_a = np.asarray(T_out, dtype=float)
    T_s = np.asarray(T_sink, dtype=float)
    if p.c is not None and len(p.c) == 4:
        c = p.c
        cop = c[0] + c[1] * T_a + c[2] * T_s + c[3] * T_a * T_s
        return np.maximum(cop, 0.1)
    # Legacy fallback: COP = capacity / power
    q = predict_capacity(T_a, T_s, p)
    pel = predict_power(T_a, T_s, p)
    return q / pel


def fit_ashp_maps(
    T_out: np.ndarray,
    T_sink: np.ndarray,
    Q_meas_kwh: np.ndarray,
    P_meas_kwh: np.ndarray,
    dt_h: float = 0.5,
    apply_high_load_filter: bool = False,
) -> ASHPParams:
    """Fit ASHP COP map (and legacy capacity map) from measured data.

    Primary output is a direct COP map fitted from back-calculated
    heat / measured electricity.  Legacy capacity (``a``) and power (``b``)
    coefficients are also fitted for backward compatibility.

    Parameters
    ----------
    T_out, T_sink : outdoor air temperature and sink-proxy arrays [°C].
    Q_meas_kwh : measured condenser heat per interval [kWh] (may be unknown;
                 pass NaN to skip capacity / COP fitting).
    P_meas_kwh : measured ASHP electrical energy per interval [kWh].
    dt_h : interval length in hours (default 0.5).

    Returns
    -------
    ASHPParams with fitted COP (``c``), capacity (``a``), and power (``b``)
    coefficients.
    """
    T_a = np.asarray(T_out, dtype=float)
    T_s = np.asarray(T_sink, dtype=float)
    P_meas = np.asarray(P_meas_kwh, dtype=float)

    # Mask: only intervals where ASHP was running at substantial load.
    valid = np.isfinite(P_meas) & (P_meas > 0.05) & np.isfinite(T_a) & np.isfinite(T_s)
    
    if apply_high_load_filter and valid.sum() > 50:
        p75 = np.percentile(P_meas[valid], HIGH_LOAD_PERCENTILE)
        mask = valid & (P_meas >= p75)
    else:
        mask = valid

    params = ASHPParams()

    # --- COP map fitting (primary) -----------------------------------------
    Q_meas = np.asarray(Q_meas_kwh, dtype=float) if Q_meas_kwh is not None else np.full_like(P_meas, np.nan)
    mask_q = mask & np.isfinite(Q_meas) & (Q_meas > 0.01)

    if mask_q.sum() > 20:
        T_a_q, T_s_q = T_a[mask_q], T_s[mask_q]
        # COP = Q / P (both in kWh units, so ratio is dimensionless)
        COP_meas = Q_meas[mask_q] / np.maximum(P_meas[mask_q], 1e-3)
        # Clip implausible COPs before fitting
        COP_meas = np.clip(COP_meas, 0.5, 8.0)

        Xc = np.column_stack([np.ones(len(T_a_q)), T_a_q, T_s_q, T_a_q * T_s_q])
        c_ols, _, _, _ = np.linalg.lstsq(Xc, COP_meas, rcond=None)
        c_lo = np.array([-10.0, -0.5, -0.5, -0.02])
        c_hi = np.array([20.0,   0.5,  0.5,  0.02])
        c_init = np.clip(c_ols, c_lo + 1e-6, c_hi - 1e-6)

        def cop_residuals(c):
            pred = Xc @ c
            pred = np.maximum(pred, 0.1)
            return pred - COP_meas

        res_c = least_squares(
            cop_residuals, c_init,
            bounds=(c_lo, c_hi),
            loss="soft_l1",
        )
        params.c = res_c.x
        logger.info("ASHP COP map coefficients: %s", params.c)
    else:
        # Default COP map when no Q data available
        params.c = np.array([3.0, 0.05, -0.02, 0.0])
        logger.info("No Q data for COP fitting; using default COP map.")

    # --- Legacy capacity map fitting (a coefficients) ----------------------
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
        logger.info("ASHP capacity map coefficients (legacy): %s", params.a)
    else:
        avg_cop = 3.0
        params.a = np.array([8.0, 0.1, -0.05, 0.0])
        logger.info("ASHP capacity estimated from defaults.")

    # --- Legacy power map fitting (b coefficients) -------------------------
    T_a_f, T_s_f, P_f = T_a[mask], T_s[mask], P_meas[mask]
    if len(P_f) > 10:
        P_kw = P_f / dt_h
        X = np.column_stack([np.ones(len(T_a_f)), T_a_f, T_s_f, T_a_f * T_s_f])
        b_ols, _, _, _ = np.linalg.lstsq(X, P_kw, rcond=None)
        b_lo = np.array([-20.0, -0.5, -0.5, -0.02])
        b_hi = np.array([20.0,   0.5,  0.5,  0.02])
        b_init = np.clip(b_ols, b_lo + 1e-6, b_hi - 1e-6)

        def power_residuals(b):
            pred = X @ b
            pred = np.maximum(pred, 0.1)
            return pred - P_kw

        res_b = least_squares(
            power_residuals, b_init,
            bounds=(b_lo, b_hi),
            loss="soft_l1",
        )
        params.b = res_b.x
        logger.info("ASHP power map coefficients (legacy): %s", params.b)

    return params
