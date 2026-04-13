"""
ST_fitting.st_model – Pure prediction functions for the solar-thermal subsystem.

These functions are imported by both the ST fitting pipeline and the tank
model (``src.tank_model``).  To avoid circular imports this module depends
**only** on numpy — no imports from ``src/`` or other ``ST_fitting/``
sub-modules.

Models
------
Q_sol_kwh  = max(0, q0 + q1*GTI + q2*T_bottom + q3*T_out)
                                                — interval ST energy [kWh]
T_flow     = b0 + b1 * T_bottom + b2 * GTI    — bilinear flow-temperature map [°C]

Dynamic node allocation
-----------------------
Each node receives ST heat proportional to (T_flow − T_node) clipped to ≥ 0:
    ΔT_i = max(T_flow − T_i, 0)
    D    = max(Σ ΔT_i, 1)            (floor of 1 prevents division by zero)
    w_i  = ΔT_i / D
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Interval energy prediction
# ---------------------------------------------------------------------------

def predict_q_sol_kwh(
    gti_wm2: float,
    q0_kwh: float,
    q1_kwh_per_wm2: float,
    *,
    t_bottom_c: float = 0.0,
    t_out_c: float = 0.0,
    q2_kwh_per_c: float = 0.0,
    q3_kwh_per_c: float = 0.0,
) -> float:
    """Predict ST heat delivered to the tank for one interval [kWh].

    Parameters
    ----------
    gti_wm2        : incident global tilted irradiance [W/m²].
    q0_kwh         : regression intercept [kWh].
    q1_kwh_per_wm2 : regression slope [kWh / (W/m²)].
    t_bottom_c     : bottom-node temperature [°C].
    t_out_c        : outdoor ambient temperature [°C].
    q2_kwh_per_c   : coefficient on bottom temperature [kWh / °C].
    q3_kwh_per_c   : coefficient on outdoor temperature [kWh / °C].
    """
    return max(
        0.0,
        q0_kwh
        + q1_kwh_per_wm2 * gti_wm2
        + q2_kwh_per_c * t_bottom_c
        + q3_kwh_per_c * t_out_c,
    )


# ---------------------------------------------------------------------------
# Flow-temperature prediction
# ---------------------------------------------------------------------------

def predict_t_flow_c(
    t_bottom_c: float,
    gti_wm2: float,
    b0_c: float,
    b1: float,
    b2_c_per_wm2: float,
) -> float:
    """Bilinear (no interaction) ST flow-temperature model [°C].

    T_flow = b0 + b1 * T_bottom + b2 * GTI

    Parameters
    ----------
    t_bottom_c    : current bottom-node temperature [°C].
    gti_wm2       : incident GTI [W/m²].
    b0_c          : intercept [°C].
    b1            : coefficient on T_bottom [dimensionless].
    b2_c_per_wm2  : coefficient on GTI [°C / (W/m²)].
    """
    return b0_c + b1 * t_bottom_c + b2_c_per_wm2 * gti_wm2


# ---------------------------------------------------------------------------
# Dynamic node-weight allocation
# ---------------------------------------------------------------------------

def st_node_weights_from_tflow(
    t_nodes_c: np.ndarray,
    t_flow_c: float,
) -> np.ndarray:
    """Return per-node ST heat allocation weights based on thermal headroom.

    Each node's raw weight is the positive part of (T_flow − T_node):
        ΔT_i = max(T_flow − T_i, 0)
    Weights are normalised by D = max(Σ ΔT_i, 1) so that they sum to ≤ 1.
    When all nodes are at or above T_flow (D stays at the floor of 1),
    all weights are zero and no ST heat is injected.

    Parameters
    ----------
    t_nodes_c : array of shape (4,) — node temperatures bottom→top [°C].
    t_flow_c  : predicted ST flow temperature [°C].

    Returns
    -------
    weights : array of shape (4,), each ≥ 0.
    """
    dt = np.maximum(t_flow_c - np.asarray(t_nodes_c, dtype=float), 0.0)
    D = max(float(dt.sum()), 1.0)
    return dt / D


# ---------------------------------------------------------------------------
# Activation gate
# ---------------------------------------------------------------------------

def solar_active(
    gti_wm2: float,
    t_bottom_c: float,
    gti_min_wm2: float = 180.0,
    t_bottom_max_c: float = 55.0,
) -> bool:
    """Return True when the ST system should be modelled as active.

    Active when irradiance is at or above *gti_min_wm2* **and** the
    bottom node is below *t_bottom_max_c*.

    Parameters
    ----------
    gti_wm2        : current GTI [W/m²].
    t_bottom_c     : current bottom-node temperature [°C].
    gti_min_wm2    : minimum GTI for activation [W/m²].
    t_bottom_max_c : bottom-node saturation temperature [°C].
    """
    return gti_wm2 >= gti_min_wm2 and t_bottom_c < t_bottom_max_c
