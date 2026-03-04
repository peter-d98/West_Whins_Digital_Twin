"""
4-node DHW tank grey-box model.

States:  T_b, T_m, T_mh, T_t  (bottom, mid, mid-hi, top).
Inputs per interval:
  - Q_ST   : solar-thermal heat delivered [kWh]
  - Q_ASHP : ASHP condenser heat delivered [kWh]
  - Q_imm  : immersion heater heat [kWh]
  - T_amb  : ambient (plant room) temperature [°C]

The tank is 550 L split into 4 equal-volume nodes (137.5 L each).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Physical constants
RHO = 1000.0   # kg/m³
CP  = 4.186     # kJ/(kg·K)
NODE_VOL_L = 550.0 / 4.0   # litres per node
NODE_MASS = NODE_VOL_L * RHO / 1000.0   # kg  (137.5 kg)
NODE_CAP  = NODE_MASS * CP  # kJ/K  (≈575.3)


@dataclass
class TankParams:
    """Contains all the learnable parameters of the grey-box model.
    These are the parameters that will be optimised to fit the model to real data.

    UA_loss : per-node UA to ambient [kW/K] (4 values, bottom→top).
    UA_adj  : adjacent-node conductance [kW/K] (3 values: b-m, m-mh, mh-t).
    f_st    : fraction of ST heat to each node (4 values, should sum ≈1).
    f_ashp  : fraction of ASHP heat to each node (4 values).
    f_imm   : fraction of immersion heat to each node (4 values).
    mix_coeff : draw-induced mixing coefficient [kW/K].
    draw_ua : per-node UA to cold mains water [kW/K] (4 values).
    T_mains : cold mains water temperature [°C].
    """
    #default values are physically informed intitial guesses
    UA_loss: np.ndarray = field(default_factory=lambda: np.array([0.003, 0.002, 0.002, 0.003]))
    UA_adj:  np.ndarray = field(default_factory=lambda: np.array([0.05, 0.05, 0.05]))
    f_st:    np.ndarray = field(default_factory=lambda: np.array([0.0, 0.3, 0.5, 0.2]))
    f_ashp:  np.ndarray = field(default_factory=lambda: np.array([0.1, 0.4, 0.3, 0.2]))
    f_imm:   np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.2, 0.8]))
    mix_coeff: float = 0.01
    draw_ua: np.ndarray = field(default_factory=lambda: np.array([0.01, 0.005, 0.002, 0.001]))
    T_mains: float = 10.0  # cold mains water temperature [°C]

    def to_vector(self) -> np.ndarray:
        """Flatten all parameters to a 1-D vector for optimisation."""
        return np.concatenate([
            self.UA_loss,       # 4
            self.UA_adj,        # 3
            self.f_st,          # 4
            self.f_ashp,        # 4
            self.f_imm,         # 4
            [self.mix_coeff],   # 1
            self.draw_ua,       # 4
        ])                      # total = 24

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "TankParams":
        """Reverses to_vector, reconstructs a TankParams instance with array slicing"""
        p = cls()
        p.UA_loss    = v[0:4]
        p.UA_adj     = v[4:7]
        p.f_st       = v[7:11]
        p.f_ashp     = v[11:15]
        p.f_imm      = v[15:19]
        p.mix_coeff  = float(v[19])
        p.draw_ua    = v[20:24]
        return p

    @staticmethod
    def lower_bounds() -> np.ndarray:
        return np.array([
            0, 0, 0, 0,               # UA_loss
            0, 0, 0,                   # UA_adj
            0, 0, 0, 0,               # f_st
            0, 0, 0, 0,               # f_ashp
            0, 0, 0, 0,               # f_imm
            0,                         # mix_coeff
            0, 0, 0, 0,               # draw_ua
        ], dtype=float)

    @staticmethod
    def upper_bounds() -> np.ndarray:
        return np.array([
            0.05, 0.05, 0.05, 0.05,   # UA_loss
            0.5, 0.5, 0.5,            # UA_adj
            1, 1, 1, 1,               # f_st
            1, 1, 1, 1,               # f_ashp
            1, 1, 1, 1,               # f_imm
            0.2,                       # mix_coeff
            0.1, 0.1, 0.1, 0.1,       # draw_ua
        ], dtype=float)


def tank_step(
    T: np.ndarray,
    Q_st_kwh: float,
    Q_ashp_kwh: float,
    Q_imm_kwh: float,
    T_amb: float,
    params: TankParams,
    dt_s: float = 1800.0,
) -> np.ndarray:
    """Advance the 4-node tank by one time step (Euler forward).

    Parameters
    ----------
    T : array of shape (4,) — current temperatures [°C].
    Q_st_kwh, Q_ashp_kwh, Q_imm_kwh : heat inputs this interval [kWh].
    T_amb : ambient temperature [°C].
    params : TankParams instance.
    dt_s : time-step in seconds (default 1800 = 30 min).

    Returns
    -------
    T_new : updated temperatures (4,) [°C].
    """
    T = np.array(T, dtype=float)
    T_new = T.copy()
    T_mains = params.T_mains

    # Convert kWh → kJ for the interval
    Q_st_kj  = Q_st_kwh * 3600.0
    Q_ashp_kj = Q_ashp_kwh * 3600.0
    Q_imm_kj  = Q_imm_kwh * 3600.0

    for i in range(4):
        # Heat input to this node [kJ] e.g. if f_st[3]=0.2, top node gets 20% of ST heat input
        dQ = (params.f_st[i] * Q_st_kj
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

        # Draw-induced mixing (tendency toward neighbour average)
        mix = 0.0
        if i > 0:
            mix += params.mix_coeff * (T[i - 1] - T[i]) * dt_s
        if i < 3:
            mix += params.mix_coeff * (T[i + 1] - T[i]) * dt_s

        # Draw loss — cold mains water replacement [kJ]
        draw_loss = params.draw_ua[i] * (T[i] - T_mains) * dt_s

        dT = (dQ - loss + cond + mix - draw_loss) / NODE_CAP
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
) -> np.ndarray:
    """Run the tank model over N time steps.

    Parameters
    ----------
    T0 : initial temperatures (4,).
    Q_st, Q_ashp, Q_imm : heat input arrays of shape (N,) [kWh per step].
    T_amb : ambient temperature array of shape (N,) [°C].
    params : TankParams.
    dt_s : time-step seconds.

    Returns
    -------
    T_hist : array (N+1, 4) — temperatures at each step (including T0).
    """
    N = len(Q_st)
    T_hist = np.zeros((N + 1, 4))
    T_hist[0] = T0

    # Each step feeds the output of the previous step(T_hist[k]) as the input to the next (T_hist[k+1]).
    for k in range(N):
        T_hist[k + 1] = tank_step(
            T_hist[k],
            float(Q_st[k]),
            float(Q_ashp[k]),
            float(Q_imm[k]),
            float(T_amb[k]),
            params,
            dt_s,
        )
    return T_hist
