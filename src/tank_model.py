"""
4-node DHW tank grey-box model.

States:  T_b, T_m, T_mh, T_t  (bottom, mid, mid-hi, top).
Inputs per interval:
  - Q_ST   : solar-thermal heat delivered [kWh]
  - Q_ASHP : ASHP condenser heat delivered [kWh]
  - Q_imm  : immersion heater heat [kWh]
  - T_amb  : ambient (plant room) temperature [°C]

Tank geometry (550 L total, non-uniform nodes):
  - Bottom node : 170 L   → 711.62 kJ/K
  - Mid, Mid-Hi, Top : 380 L / 3 ≈ 126.67 L each → 530.21 kJ/K each
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Physical constants
_RHO = 1.0        # kg/L
_CP  = 4.186      # kJ/(kg·K)
NODE_CAP = np.array([
    170.0         * _RHO * _CP,   # bottom  → 711.62 kJ/K
    (380.0 / 3.0) * _RHO * _CP,   # mid     → 530.21 kJ/K
    (380.0 / 3.0) * _RHO * _CP,   # mid-hi  → 530.21 kJ/K
    (380.0 / 3.0) * _RHO * _CP,   # top     → 530.21 kJ/K
])  # shape (4,), kJ/K, bottom→top


@dataclass
class TankParams:
    """Contains all the learnable parameters of the grey-box model.
    These are the parameters that will be optimised to fit the model to real data.

    UA_loss : per-node UA to ambient [kW/K] (4 values, bottom→top).
    UA_adj  : adjacent-node conductance [kW/K] (3 values: b-m, m-mh, mh-t).
    f_st    : fraction of ST heat to each node (4 values, should sum ≈1).
    f_ashp  : fraction of ASHP heat to each node (4 values).
    f_imm   : fraction of immersion heat to each node (4 values).
    """
    #default values are physically informed intitial guesses
    UA_loss: np.ndarray = field(default_factory=lambda: np.array([0.003, 0.002, 0.002, 0.003]))
    UA_adj:  np.ndarray = field(default_factory=lambda: np.array([0.05, 0.05, 0.05]))
    f_st:    np.ndarray = field(default_factory=lambda: np.array([0.0, 0.3, 0.5, 0.2]))
    f_ashp:  np.ndarray = field(default_factory=lambda: np.array([0.1, 0.4, 0.3, 0.2]))
    f_imm:   np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.2, 0.8]))

    def to_vector(self) -> np.ndarray:
        """Flatten all parameters to a 1-D vector for optimisation."""
        return np.concatenate([
            self.UA_loss,       # 4
            self.UA_adj,        # 3
            self.f_st,          # 4
            self.f_ashp,        # 4
            self.f_imm,         # 4
        ])                      # total = 19

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "TankParams":
        """Reverses to_vector, reconstructs a TankParams instance with array slicing."""
        p = cls()
        p.UA_loss    = v[0:4]
        p.UA_adj     = v[4:7]
        p.f_st       = v[7:11]
        p.f_ashp     = v[11:15]
        p.f_imm      = v[15:19]
        return p

    @staticmethod
    def lower_bounds() -> np.ndarray:
        return np.array([
            0, 0, 0, 0,               # UA_loss
            0, 0, 0,                   # UA_adj
            0, 0, 0, 0,               # f_st
            0, 0, 0, 0,               # f_ashp
            0, 0, 0, 0,               # f_imm
        ], dtype=float)

    @staticmethod
    def upper_bounds() -> np.ndarray:
        return np.array([
            0.05, 0.05, 0.05, 0.05,   # UA_loss
            0.5, 0.5, 0.5,            # UA_adj
            1, 1, 1, 1,               # f_st
            1, 1, 1, 1,               # f_ashp
            1, 1, 1, 1,               # f_imm
        ], dtype=float)


def tank_step(
    T: np.ndarray,
    Q_st_kwh: float,
    Q_ashp_kwh: float,
    Q_imm_kwh: float,
    T_amb: float,
    params: TankParams,
    dt_s: float = 1800.0,
    *,
    f_st_ext: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Advance the 4-node tank by one time step (Euler forward).

    Parameters
    ----------
    T : array of shape (4,) — current temperatures [°C].
    Q_st_kwh : ST heat this interval [kWh].
    Q_ashp_kwh, Q_imm_kwh : heat inputs this interval [kWh].
    T_amb : ambient temperature [°C].
    params : TankParams instance.
    dt_s : time-step in seconds (default 1800 = 30 min).
    f_st_ext : Optional array of shape (4,) — external ST node allocation
        weights.  When provided, overrides ``params.f_st`` for this step.
        This allows callers to pre-compute dynamic node allocation outside
        the tank model (e.g., from solar regressions).

    Returns
    -------
    T_new : updated temperatures (4,) [°C].
    """
    T = np.array(T, dtype=float)
    T_new = T.copy()

    # Determine ST node allocation weights
    if f_st_ext is not None:
        f_st = np.asarray(f_st_ext, dtype=float)
    else:
        f_st = params.f_st

    # Convert kWh → kJ for the interval
    Q_st_kj   = Q_st_kwh * 3600.0
    Q_ashp_kj = Q_ashp_kwh * 3600.0
    Q_imm_kj  = Q_imm_kwh * 3600.0

    for i in range(4):
        # Heat input to this node [kJ]
        dQ = (f_st[i] * Q_st_kj
              + params.f_ashp[i] * Q_ashp_kj
              + params.f_imm[i] * Q_imm_kj)

        # Loss to ambient [kJ] = UA [kW/K] × ΔT [K] × dt [s]
        loss = params.UA_loss[i] * (T[i] - T_amb) * dt_s

        # Adjacent-node conduction [kJ]
        cond = 0.0
        if i > 0:
            cond += params.UA_adj[i - 1] * (T[i - 1] - T[i]) * dt_s
        if i < 3:
            cond += params.UA_adj[i] * (T[i + 1] - T[i]) * dt_s

        dT = (dQ - loss + cond) / NODE_CAP[i]
        T_new[i] = T[i] + dT

    # Enforce plausible bounds
    T_new = np.clip(T_new, 5.0, 95.0)
    return T_new


def simulate(
    T0: np.ndarray,
    Q_st: np.ndarray,
    Q_ashp: np.ndarray,
    Q_imm: np.ndarray,
    T_amb: np.ndarray,
    params: TankParams,
    dt_s: float = 1800.0,
    *,
    f_st_ext: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Run the tank model over N time steps.

    Parameters
    ----------
    T0 : initial temperatures (4,).
    Q_st, Q_ashp, Q_imm : heat input arrays of shape (N,) [kWh per step].
    T_amb : ambient temperature array of shape (N,) [°C].
    params : TankParams.
    dt_s : time-step seconds.
    f_st_ext : Optional array of shape (N, 4) — external ST node allocation
        weights per step.  When provided, overrides ``params.f_st`` at each
        step.  This allows callers to pre-compute dynamic node allocation
        outside the tank model (e.g., from solar regressions).

    Returns
    -------
    T_hist : array (N+1, 4) — temperatures at each step (including T0).
    """
    N = len(Q_st)
    T_hist = np.zeros((N + 1, 4))
    T_hist[0] = T0

    for k in range(N):
        kw: dict = {}
        if f_st_ext is not None:
            kw["f_st_ext"] = f_st_ext[k]
        T_hist[k + 1] = tank_step(
            T_hist[k],
            float(Q_st[k]),
            float(Q_ashp[k]),
            float(Q_imm[k]),
            float(T_amb[k]),
            params,
            dt_s,
            **kw,
        )
    return T_hist
