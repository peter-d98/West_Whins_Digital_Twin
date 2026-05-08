"""Model-Predictive Controller for the DHW tank.

Drop-in controller compatible with ``src.control.simulator.simulate()``:
``MPCController.__call__(ctx, T_state) -> bool``.

Decision: fire ASHP override (snap mid→55, mid_hi→55.5, top→56) or not,
each 30-min slot. Solar override is handled automatically by the
simulator. Legionella firings are pre-planned by ``plan_legionella_overrides``
and passed to ``simulate(legionella_overrides=...)``.

Algorithm (true receding-horizon, replan every step, commit slot 1):

  1. Build forecast horizon of length H steps from current step k.
  2. Build a deterministic hedged DHW scale = 1 + n_sigma · CV[month]
     so the planner sees a worst-case-ish demand profile.
  3. Greedy slot-insertion:
        schedule = [False] * H
        rollout = forecast(schedule)
        while min(rollout.mid) < comfort_mid_min:
            v = first step with mid < comfort_mid_min
            cands = sorted [0, v) by ascending price[j] / COP(T_amb[j], 55)
            for j in cands until rollout feasible up to v:
                schedule[j] = True
                rollout = forecast(schedule)
                if rollout still violates at <= v: revert and try next
                else: break
  4. Final reduction pass: walk schedule; for each True, try removing —
     keep removal iff still feasible.
  5. Commit schedule[0].

Each MPC step caches its own forecast price / T_amb / illum slices so the
hot loop is tight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.ashp_model import ASHPParams, predict_cop
from src.control.simulator import (
    LegionellaScheduler,
    SimContext,
    simulate,
)
from src.tank_model import TankParams


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class MPCConfig:
    horizon_steps: int = 96         # 48 h
    comfort_mid_min: float = 40.0   # hard lower bound on the comfort node
    comfort_node: int = 1           # node index used for the comfort floor
                                    # (1 = mid, 3 = top)
    ashp_setpoint: float = 55.0
    solar_thresh: float = 200.0
    solar_setpoint: float = 60.0
    t_mains: float = 10.0
    dhw_mode: str = "bottom_only"
    n_sigma_hedge: float = 1.0      # multiplier on CV for forecast hedge
    replan_every: int = 1           # 1 = true receding horizon


# ---------------------------------------------------------------------------
# Legionella pre-scheduler (open-loop optimal within deadlines)
# ---------------------------------------------------------------------------
def plan_legionella_overrides(
    times: pd.DatetimeIndex,
    price_p_per_kwh: pd.Series,
    *,
    period_days: int = 7,
    init_days_since_last: int = 0,
    steps_per_day: int = 48,
) -> np.ndarray:
    """Pick the cheapest 30-min slot inside each rolling 7-day window.

    Returns a length-N boolean array. The first window starts at
    step 0 with deadline = (period_days - init_days_since_last) * 48.
    Subsequent windows start at the previous fire + 1 step.
    """
    n = len(times)
    out = np.zeros(n, dtype=bool)
    period = period_days * steps_per_day
    init_off = int(init_days_since_last) * steps_per_day
    p = price_p_per_kwh.reindex(times).ffill().bfill().to_numpy()

    # Tile the slice into back-to-back `period`-sized windows. The first
    # window is shortened by `init_off` (the time elapsed since the last
    # cycle before the slice started). A window contributes one fire iff
    # it lies fully inside the slice — partial trailing windows defer to
    # the next slice.
    cursor = -init_off
    while True:
        win_start = max(0, cursor)
        win_end = cursor + period
        if win_end > n:
            break
        slot = win_start + int(np.argmin(p[win_start:win_end]))
        out[slot] = True
        cursor = win_end

    return out


# ---------------------------------------------------------------------------
# Lightweight hedged-forecast rollout
# ---------------------------------------------------------------------------
class _Forecaster:
    """Wraps repeated calls to ``simulate()`` over a planning horizon.

    Caches the slice arrays for one MPC step so repeated rollouts (one per
    candidate schedule) reuse them.
    """

    def __init__(
        self,
        times: pd.DatetimeIndex,
        T_amb: np.ndarray,
        illum: np.ndarray,
        profile_by_month: dict[int, np.ndarray],
        cv_by_month: dict[int, float],
        params: TankParams,
        ashp_params: ASHPParams,
        leg: LegionellaScheduler,
        leg_overrides: np.ndarray,
        cfg: MPCConfig,
    ) -> None:
        self.times = times
        self.T_amb = T_amb
        self.illum = illum
        self.profile_by_month = profile_by_month
        self.params = params
        self.ashp_params = ashp_params
        self.leg = leg
        self.leg_overrides = leg_overrides
        self.cfg = cfg
        # Hedged daily_scale: deterministic, broadcast 1 + n_sigma * CV[month]
        scale = np.ones(len(times), dtype=float)
        for k, ts in enumerate(times):
            cv = float(cv_by_month.get(int(ts.month), 0.4))
            scale[k] = 1.0 + cfg.n_sigma_hedge * cv
        self.daily_scale = scale
        # Pre-compute per-step expected per-fire cost-of-energy proxy:
        # price / COP(T_amb, T_sink≈55). Lower = cheaper place to fire.
        T_sink = np.full_like(T_amb, cfg.ashp_setpoint + 0.5)
        cop = predict_cop(T_amb, T_sink, ashp_params)
        cop = np.maximum(cop, 1.0)
        self._cop = cop

    def cost_metric(self, price_slice: np.ndarray) -> np.ndarray:
        """Per-slot 'cost per delivered kWh' = price/COP (p/kWh / dimensionless)."""
        return price_slice / self._cop

    def rollout(self, T0: np.ndarray, schedule: np.ndarray):
        """Run one forecast given a schedule. Returns (T_hist, E_ashp)."""
        sched = schedule  # captured by closure

        def _ctrl(ctx: SimContext, T_state: np.ndarray) -> bool:
            return bool(sched[ctx.k])

        res = simulate(
            T0=T0, times=self.times, T_amb=self.T_amb, solar_illum=self.illum,
            profile_by_month=self.profile_by_month, params=self.params,
            ashp_params=self.ashp_params, controller=_ctrl,
            cv_by_month={m: 0.0 for m in range(1, 13)},  # noise off
            daily_scale=self.daily_scale,
            solar_thresh=self.cfg.solar_thresh,
            solar_setpoint=self.cfg.solar_setpoint,
            ashp_setpoint=self.cfg.ashp_setpoint,
            t_mains=self.cfg.t_mains,
            dhw_mode=self.cfg.dhw_mode,
            legionella=self.leg,
            legionella_overrides=self.leg_overrides,
            safety_mid_floor=-np.inf,  # planner detects infeasibility itself
        )
        return res


# ---------------------------------------------------------------------------
# MPC controller
# ---------------------------------------------------------------------------
class MPCController:
    """Receding-horizon greedy MPC controller.

    Plug-in compatible with :func:`src.control.simulator.simulate`. Pass
    ``MPCController(...)`` as the ``controller=`` argument and pass the
    same ``legionella_overrides`` you constructed via
    :func:`plan_legionella_overrides` to the outer ``simulate`` call.
    """

    def __init__(
        self,
        *,
        times: pd.DatetimeIndex,
        T_amb: np.ndarray,
        solar_illum: np.ndarray,
        price_p_per_kwh: pd.Series,
        profile_by_month: dict[int, np.ndarray],
        cv_by_month: dict[int, float],
        params: TankParams,
        ashp_params: ASHPParams,
        leg: LegionellaScheduler,
        leg_overrides: np.ndarray,
        cfg: Optional[MPCConfig] = None,
    ) -> None:
        self.times = times
        self.T_amb = np.asarray(T_amb, dtype=float)
        self.solar_illum = np.asarray(solar_illum, dtype=float)
        self.price = price_p_per_kwh.reindex(times).ffill().bfill().to_numpy()
        self.profile_by_month = profile_by_month
        self.cv_by_month = cv_by_month
        self.params = params
        self.ashp_params = ashp_params
        self.leg = leg
        self.leg_overrides = np.asarray(leg_overrides, dtype=bool)
        self.cfg = cfg or MPCConfig()
        self._N = len(times)
        # Plan cache: (k_start, schedule) — only first slot is committed.
        self._cached_step: int = -1
        self._cached_decision: bool = False
        # Diagnostic counters
        self.n_replans = 0
        self.n_rollouts = 0

    # ------------------------------------------------------------------
    # The entry point used by simulate()
    # ------------------------------------------------------------------
    def __call__(self, ctx: SimContext, T_state: np.ndarray) -> bool:
        # Replan only when needed (default: every step).
        if self._cached_step != ctx.k or (ctx.k % self.cfg.replan_every == 0):
            self._cached_decision = self._plan_and_commit(ctx, T_state)
            self._cached_step = ctx.k
            self.n_replans += 1
        return self._cached_decision

    # ------------------------------------------------------------------
    def _plan_and_commit(self, ctx: SimContext, T_state: np.ndarray) -> bool:
        k0 = ctx.k
        H = min(self.cfg.horizon_steps, self._N - k0)
        if H <= 0:
            return False

        sl = slice(k0, k0 + H)
        fc = _Forecaster(
            times=self.times[sl],
            T_amb=self.T_amb[sl],
            illum=self.solar_illum[sl],
            profile_by_month=self.profile_by_month,
            cv_by_month=self.cv_by_month,
            params=self.params,
            ashp_params=self.ashp_params,
            leg=self.leg,
            leg_overrides=self.leg_overrides[sl].copy(),
            cfg=self.cfg,
        )
        cost_metric = fc.cost_metric(self.price[sl])

        schedule = np.zeros(H, dtype=bool)
        T0 = T_state.copy()

        # ---- Greedy: insert until comfort satisfied ----
        max_inserts = H  # safety cap
        node = self.cfg.comfort_node
        for _ in range(max_inserts):
            res = fc.rollout(T0, schedule); self.n_rollouts += 1
            mid_seq = res.T_hist[1:, node]
            below = np.where(mid_seq < self.cfg.comfort_mid_min)[0]
            if len(below) == 0:
                break
            v = int(below[0])  # first violation step
            # Candidate insertion slots: in [0, v], not already on.
            window_end = v + 1  # allow inserting AT the violation slot too
            cands = [j for j in range(0, window_end) if not schedule[j]]
            if not cands:
                # Cannot insert anywhere — accept infeasibility, return on now.
                schedule[0] = True
                break
            # Try cheapest first; accept the first that strictly improves
            # min mid over [0, v+1) (so we make progress even if not yet feasible).
            cands.sort(key=lambda j: cost_metric[j])
            improved = False
            base_min = mid_seq[: v + 1].min()
            for j in cands:
                schedule[j] = True
                res2 = fc.rollout(T0, schedule); self.n_rollouts += 1
                new_min = res2.T_hist[1:, node][: v + 1].min()
                if new_min > base_min + 1e-6:
                    improved = True
                    break
                schedule[j] = False
            if not improved:
                # No slot helps (rare); force the cheapest and continue.
                schedule[cands[0]] = True

        # ---- Final reduction pass (drop any redundant fires) ----
        on_slots = list(np.where(schedule)[0])
        # Don't try to drop legionella overrides (they're not in `schedule`).
        for j in on_slots:
            schedule[j] = False
            res = fc.rollout(T0, schedule); self.n_rollouts += 1
            if res.T_hist[1:, node].min() < self.cfg.comfort_mid_min - 1e-6:
                schedule[j] = True  # restore

        return bool(schedule[0])
