"""Closed-loop tank simulator for predictive-control studies.

Refactored from ``scripts/validate_window.py``: the physics rules are
identical, but the ASHP firing decision is delegated to a controller
callback ``decide_ashp(t, T_state, ctx) -> bool``. Solar override and
DHW dynamics remain automatic. A weekly legionella cycle is layered on
top via ``LegionellaScheduler``.

Per-step accounting attaches:
  * `Q_ashp_kwh` — heat injected to the upper-3 nodes by ASHP overrides
                   (Σ m·Cp·ΔT for each node forced upward).
  * `Q_st_kwh`   — heat injected by solar override (4 nodes).
  * `Q_imm_kwh`  — heat injected by the legionella immersion override.
  * `E_ashp_kwh` — electricity = Q_ashp / COP(T_amb, T_sink).
  * `E_imm_kwh`  — electricity = Q_imm  (immersion ≈ 100 % efficient).

Coordinate convention: nodes are bottom→top = [0,1,2,3].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.tank_model import NODE_CAP, NODE_VOLS, TankParams, dhw_step, tank_step
from src.ashp_model import ASHPParams, predict_cop


DT_S = 1800.0  # 30-min default

# Indexing helpers
B, M, MH, T = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# Sim context passed to controllers
# ---------------------------------------------------------------------------
@dataclass
class SimContext:
    """Snapshot of conditions visible to a controller at decision time."""
    t: pd.Timestamp
    k: int                 # step index
    T_amb: float           # ambient now
    illum: float           # solar illuminance now
    days_since_legionella: int


@dataclass
class SimResult:
    times: pd.DatetimeIndex
    T_hist: np.ndarray            # (N+1, 4)
    solar_flags: np.ndarray       # (N,) bool
    ashp_flags: np.ndarray        # (N,) bool
    imm_flags: np.ndarray         # (N,) bool — legionella firings
    V_draws: np.ndarray           # (N,) litres applied
    Q_ashp_kwh: np.ndarray        # (N,) heat injected
    Q_st_kwh: np.ndarray          # (N,)
    Q_imm_kwh: np.ndarray         # (N,)
    E_ashp_kwh: np.ndarray        # (N,) electricity drawn by ASHP
    E_imm_kwh: np.ndarray         # (N,) electricity drawn by immersion


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------
ControllerFn = Callable[[SimContext, np.ndarray], bool]


def threshold_controller(ashp_trigger: float = 51.0) -> ControllerFn:
    """Baseline rule: fire ASHP if mid-node ≤ ``ashp_trigger`` °C."""
    def _decide(ctx: SimContext, T_state: np.ndarray) -> bool:  # noqa: ARG001
        return bool(T_state[M] <= ashp_trigger)
    return _decide


# ---------------------------------------------------------------------------
# Legionella scheduler
# ---------------------------------------------------------------------------
@dataclass
class LegionellaScheduler:
    """Ensures the upper 3 nodes hit ``hot_setpoint`` every ``period_days``.

    The scheduler exposes two methods:
      * ``can_fire(t, days_since_last)`` — controller may opt-in early.
      * ``must_fire(days_since_last)``   — deadline reached, force fire.

    For the baseline (threshold) controller we just force-fire on the
    deadline. For MPC, the schedule slot is chosen by the optimiser
    inside its planning horizon.
    """
    period_days: int = 7
    hot_setpoint: float = 65.0

    def must_fire(self, days_since_last: int) -> bool:
        return days_since_last >= self.period_days


# ---------------------------------------------------------------------------
# DHW forecast helpers
# ---------------------------------------------------------------------------
def hedged_profile(
    profile_by_month: dict[int, np.ndarray],
    cv_by_month: dict[int, float],
    n_sigma: float = 1.0,
) -> dict[int, np.ndarray]:
    """Return mean-profile inflated by ``(1 + n_sigma * CV)`` for forecasts."""
    out: dict[int, np.ndarray] = {}
    for m, arr in profile_by_month.items():
        cv = float(cv_by_month.get(m, 0.4))
        out[m] = arr * (1.0 + n_sigma * cv)
    return out


def sample_daily_scale(cv: float, rng: np.random.Generator) -> float:
    if cv <= 0.0:
        return 1.0
    sigma = float(np.sqrt(np.log(1.0 + cv * cv)))
    mu = -0.5 * sigma * sigma
    return float(rng.lognormal(mean=mu, sigma=sigma))


# ---------------------------------------------------------------------------
# Heat injected by an upper-3 setpoint snap
# ---------------------------------------------------------------------------
def _q_kwh_to_setpoint(T_before: np.ndarray, T_after: np.ndarray) -> float:
    """Energy required to move per-node from T_before to T_after [kWh].

    Uses NODE_CAP [kJ/K]. Returns only positive contributions.
    """
    dT = np.maximum(T_after - T_before, 0.0)
    kj = float(np.sum(NODE_CAP * dT))
    return kj / 3600.0  # kWh


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------
def simulate(
    T0: np.ndarray,
    times: pd.DatetimeIndex,
    T_amb: np.ndarray,
    solar_illum: np.ndarray,
    profile_by_month: dict[int, np.ndarray],
    params: TankParams,
    ashp_params: ASHPParams,
    controller: ControllerFn,
    *,
    cv_by_month: Optional[dict[int, float]] = None,
    rng: Optional[np.random.Generator] = None,
    daily_scale: Optional[np.ndarray] = None,
    solar_thresh: float = 200.0,
    solar_setpoint: float = 60.0,
    ashp_setpoint: float = 55.0,
    t_mains: float = 10.0,
    dhw_mode: str = "bottom_only",
    legionella: Optional[LegionellaScheduler] = None,
    legionella_init_days: int = 0,
    legionella_overrides: Optional[np.ndarray] = None,
    safety_mid_floor: float = 40.0,
    dt_s: float = DT_S,
) -> SimResult:
    """Run a closed-loop simulation.

    Parameters
    ----------
    daily_scale
        If given, a length-N array of pre-computed scale factors per step
        (cell value broadcast across that calendar day). Overrides RNG
        sampling — used to share seeds between baseline and MPC runs.
    legionella_overrides
        Optional length-N bool array; True forces a legionella fire at
        that step (used by the MPC planner). When None and ``legionella``
        is given, the scheduler force-fires on its deadline.
    safety_mid_floor
        Hard floor: if predicted mid-node would drop below this AND the
        controller said no, force ASHP fire anyway. Set to ``-inf`` to
        disable.
    """
    N = len(times)
    if cv_by_month is None:
        cv_by_month = {m: 0.4 for m in range(1, 13)}
    if rng is None:
        rng = np.random.default_rng(0)

    T_hist = np.zeros((N + 1, 4), dtype=float)
    T_hist[0] = np.asarray(T0, dtype=float).copy()
    solar_flags = np.zeros(N, dtype=bool)
    ashp_flags = np.zeros(N, dtype=bool)
    imm_flags = np.zeros(N, dtype=bool)
    V_draws = np.zeros(N)
    Q_ashp = np.zeros(N)
    Q_st = np.zeros(N)
    Q_imm = np.zeros(N)
    E_ashp = np.zeros(N)
    E_imm = np.zeros(N)

    solar_locked = False
    current_date = None
    s_today = 1.0
    days_since_leg = int(legionella_init_days)
    last_date = times[0].normalize()

    for k in range(N):
        ts = times[k]
        date = ts.normalize()
        if date != current_date:
            if current_date is not None:
                # advance legionella counter once per new calendar day
                days_since_leg += int((date - current_date).days)
            current_date = date
            if daily_scale is None:
                cv = float(cv_by_month.get(int(ts.month), 0.4))
                s_today = sample_daily_scale(cv, rng)
            else:
                s_today = float(daily_scale[k])

        T_prev = T_hist[k]

        # 1. Idle baseline (UA + conduction only)
        T_next = tank_step(
            T_prev, Q_st_kwh=0.0, Q_ashp_kwh=0.0, Q_imm_kwh=0.0,
            T_amb=float(T_amb[k]), params=params, dt_s=dt_s,
        )

        # 2. DHW draw — applied before heat-source decisions
        slot = ts.hour * 2 + ts.minute // 30
        mean_V = float(profile_by_month[int(ts.month)][slot])
        V_draw = s_today * mean_V
        if V_draw > 0.0:
            T_next = dhw_step(T_next, V_draw, T_mains_c=t_mains, mode=dhw_mode)
        V_draws[k] = V_draw

        # 3. Solar override (priority)
        illum = float(solar_illum[k])
        if illum < solar_thresh:
            solar_locked = False

        ctx = SimContext(t=ts, k=k, T_amb=float(T_amb[k]), illum=illum,
                         days_since_legionella=days_since_leg)
        T_after = T_next.copy()
        fired_solar = False
        fired_ashp = False
        fired_imm = False

        if (illum >= solar_thresh and not solar_locked
                and T_prev.max() < solar_setpoint):
            T_after[:] = solar_setpoint
            Q_st[k] = _q_kwh_to_setpoint(T_next, T_after)
            solar_flags[k] = True
            solar_locked = True
            fired_solar = True

        # 4. Legionella override (mandatory on deadline OR planner override)
        if not fired_solar:
            force_leg = False
            if legionella_overrides is not None and bool(legionella_overrides[k]):
                force_leg = True
            elif legionella is not None and legionella.must_fire(days_since_leg):
                force_leg = True
            if force_leg and legionella is not None:
                target = legionella.hot_setpoint
                T_leg = T_after.copy()
                T_leg[M] = max(T_leg[M], target)
                T_leg[MH] = max(T_leg[MH], target)
                T_leg[T] = max(T_leg[T], target)
                Q_imm[k] = _q_kwh_to_setpoint(T_after, T_leg)
                E_imm[k] = Q_imm[k]  # immersion ~ 100 % efficient
                T_after = T_leg
                imm_flags[k] = True
                fired_imm = True
                days_since_leg = 0

        # 5. ASHP decision (controller, plus safety floor)
        if not fired_solar:
            decision = bool(controller(ctx, T_next))
            if (not decision
                    and np.isfinite(safety_mid_floor)
                    and T_next[M] < safety_mid_floor):
                decision = True
            if decision and T_next[M] < ashp_setpoint:
                T_ash = T_after.copy()
                T_ash[M] = max(T_ash[M], ashp_setpoint)
                T_ash[MH] = max(T_ash[MH], ashp_setpoint + 0.5)
                T_ash[T] = max(T_ash[T], ashp_setpoint + 1.0)
                q = _q_kwh_to_setpoint(T_after, T_ash)
                if q > 0.0:
                    cop = float(predict_cop(
                        np.array([float(T_amb[k])]),
                        np.array([0.5 * (T_after[M] + T_after[T])]),
                        ashp_params,
                    )[0])
                    cop = max(cop, 1.0)  # safety floor on COP
                    Q_ashp[k] = q
                    E_ashp[k] = q / cop
                    T_after = T_ash
                    ashp_flags[k] = True
                    fired_ashp = True

        T_hist[k + 1] = T_after

    return SimResult(
        times=times,
        T_hist=T_hist,
        solar_flags=solar_flags,
        ashp_flags=ashp_flags,
        imm_flags=imm_flags,
        V_draws=V_draws,
        Q_ashp_kwh=Q_ashp,
        Q_st_kwh=Q_st,
        Q_imm_kwh=Q_imm,
        E_ashp_kwh=E_ashp,
        E_imm_kwh=E_imm,
    )
