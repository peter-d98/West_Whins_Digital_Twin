"""
Global_fitting.fit_global – Fit free tank parameters (UA_adj, optionally
UA_loss[0] and f_ashp mid/mid-hi fractions) using frozen priors from
UA_fitting and ASHP_fitting.

The loss function uses **one-step-ahead prediction**: at each training
interval *k*, the measured temperatures T[k−1] are taken as the initial
state, ``tank_step()`` is called once to predict T_pred[k], and the residual
is T_pred[k] − T_meas[k].  This resets the model state from measurements at
every step, preventing drift accumulation from unmodelled effects (draws,
partial-interval ASHP runs, etc.).

Physical constraints baked in as hard-coded fixed values (not optimised
by default):

- ``f_st   = [1, 0, 0, 0]`` — ST coil is physically inside the bottom node
- ``f_ashp = [0, 0, 0, 1]`` — ASHP HX circuit returns to the top node (default)
  When ``free_f_ashp=True``: ``f_ashp = [0, a, b, 1-a-b]`` is optimised as a
  proxy for hydraulic mixing during ASHP charging cycles.
- ``f_imm  = [0, 1, 0, 0]`` — immersion element is in the mid node
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from Global_fitting.config import GlobalFitConfig

# Read-only imports from main codebase
from src.ashp_model import ASHPParams, predict_capacity, predict_cop, sink_proxy
from src.data_loader import load_and_clean
from src.solar_thermal import compute_st_energy
from src.tank_model import TankParams, tank_step

logger = logging.getLogger(__name__)

# Hard-coded physical heat-distribution fractions (defaults / frozen cases)
F_ST   = np.array([1.0, 0.0, 0.0, 0.0])
F_ASHP = np.array([0.0, 0.0, 0.0, 1.0])  # overridden when free_f_ashp=True
F_IMM  = np.array([0.0, 1.0, 0.0, 0.0])


@dataclass
class GlobalFitResult:
    """Container for fitted parameters and diagnostics."""
    tank_params: TankParams
    cost_history: list[float]
    train_rmse: dict[str, float]


# ---------------------------------------------------------------------------
# Prior loading helpers
# ---------------------------------------------------------------------------

def _load_ua_priors(path: Path) -> np.ndarray:
    """Load UA_loss array from ua_fit.json."""
    if not path.exists():
        raise FileNotFoundError(
            f"UA priors file not found: {path}\n"
            f"Run the UA_fitting pipeline first:\n"
            f"  python UA_fitting/run_ua_fitting.py --csv <csv> --yaml <yaml>"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ua_loss = np.array(data["UA_loss"], dtype=float)
    if ua_loss.shape != (4,):
        raise ValueError(
            f"Expected UA_loss of shape (4,), got {ua_loss.shape} from {path}"
        )
    logger.info("Loaded UA_loss priors: %s", ua_loss.tolist())
    return ua_loss


def _load_ashp_priors(path: Path) -> ASHPParams:
    """Load ASHPParams from ashp_fit.json."""
    if not path.exists():
        raise FileNotFoundError(
            f"ASHP priors file not found: {path}\n"
            f"Run the ASHP_fitting pipeline first:\n"
            f"  python ASHP_fitting/run_ashp_fitting.py --csv <csv> --yaml <yaml>"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ashp = data["ashp"]
    params = ASHPParams(
        a=np.array(ashp["a"], dtype=float),
        b=np.array(ashp["b"], dtype=float),
    )
    logger.info("Loaded ASHP priors: a=%s, b=%s", params.a.tolist(), params.b.tolist())
    return params


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def _prepare_inputs(
    df,
    ashp_params: ASHPParams,
    dt_h: float = 0.5,
):
    """Build arrays needed for one-step-ahead prediction.

    Returns dict with keys: T_meas (N,4), Q_st, Q_ashp, Q_imm, T_amb (all N,).
    """
    T_meas = df[["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]].values

    # Solar-thermal energy
    if "st_kwh" in df.columns:
        Q_st = df["st_kwh"].fillna(0).values
    else:
        Q_st = compute_st_energy(df, dt_minutes=dt_h * 60).values

    # ASHP heat: predicted capacity × fraction of interval ASHP was on.
    # Gate logic matches ASHP_fitting/detector.py:
    #   - threshold 0.013 kWh (below = meter noise / standby, not DHW charging)
    #   - top-node temperature must be rising (diff > 1.0 °C) to exclude
    #     space-heating runs.  Top_rising is used (not mid_rising) because the
    #     coil return enters at the top — top_rising aligns with when heat is
    #     actually delivered, one step after mid rises by hydraulic displacement.
    # After detection, Q_ashp is shifted one step earlier: the cumulative
    # meter diff records energy in (k-1→k), so the heat belongs to step k-1.
    T_sink = sink_proxy(df["tank_mid_c"].values, df["tank_top_c"].values)
    T_out = df["t_out_c"].fillna(df["t_out_c"].median()).values
    cap_kw = predict_capacity(T_out, T_sink, ashp_params)
    cop    = predict_cop(T_out, T_sink, ashp_params)
    P_meas = df["ashp_inst_kwh"].fillna(0).values
    top_rising = pd.Series(df["tank_top_c"].values, index=df.index).diff().fillna(0.0).values > 1.0
    ashp_dhw_on = (P_meas > 0.013) & top_rising
    # Q_ashp = measured electrical energy × COP map (not cap × dt).
    # This scales heat input with actual measured consumption, using the map
    # only for thermal efficiency.  cap_kw retained for reference but not used.
    # NOTE - this is temporarily changed to the old logic
    Q_ashp = np.where(ashp_dhw_on, cap_kw * dt_h, 0.0)
    # Shift left by 1: P_meas[k] records energy in (k-1→k); after shift
    # Q_ashp[k] = P_meas[k+1]*COP drives step k→k+1 (correct alignment).
    Q_ashp = np.concatenate([Q_ashp[1:], [0.0]])

    # Immersion
    Q_imm = df["imm_tot_inst_kwh"].fillna(0).values

    T_amb = df["t_amb_c"].fillna(df["t_amb_c"].median()).values

    return dict(T_meas=T_meas, Q_st=Q_st, Q_ashp=Q_ashp, Q_imm=Q_imm, T_amb=T_amb)


# ---------------------------------------------------------------------------
# Core fitting
# ---------------------------------------------------------------------------

def _build_tank_params(
    x: np.ndarray,
    ua_loss_frozen: np.ndarray,
    free_ua_loss_bottom: bool,
    free_f_ashp: bool,
) -> TankParams:
    """Reconstruct a full TankParams from the free-parameter vector *x*.

    Vector layout:
      x[0:3]  = UA_adj (3 values, always free)
      x[3]    = UA_loss[0]  (only if free_ua_loss_bottom=True)
      x[-2]   = a  (f_ashp mid fraction,    only if free_f_ashp=True)
      x[-1]   = b  (f_ashp mid-hi fraction, only if free_f_ashp=True)

    When free_f_ashp=True, f_ashp = [0, a, b, 1-a-b].
    The constraint a+b ≤1 is enforced by clipping b to min(b, 1-a).
    """
    p = TankParams()
    p.UA_adj = x[0:3]

    ua_loss = ua_loss_frozen.copy()
    if free_ua_loss_bottom:
        ua_loss[0] = x[3]
    p.UA_loss = ua_loss

    p.f_st  = F_ST.copy()
    p.f_imm = F_IMM.copy()

    if free_f_ashp:
        a = float(x[-2])
        b = float(x[-1])
        b = min(b, 1.0 - a)   # enforce a + b <= 1
        p.f_ashp = np.array([0.0, a, b, 1.0 - a - b])
    else:
        p.f_ashp = F_ASHP.copy()

    return p


def _fit(
    inputs: dict,
    ua_loss_frozen: np.ndarray,
    cfg: GlobalFitConfig,
) -> GlobalFitResult:
    """Run least_squares optimisation for UA_adj (+ optionally UA_loss[0] and f_ashp)."""
    T_meas = inputs["T_meas"]
    Q_st = inputs["Q_st"]
    Q_ashp = inputs["Q_ashp"]
    Q_imm = inputs["Q_imm"]
    T_amb = inputs["T_amb"]
    N = len(Q_st)
    steps = N - 1

    # Build initial guess and bounds
    x0_parts = [np.array([0.05, 0.05, 0.05])]  # UA_adj
    lb_parts = [np.full(3, cfg.ua_adj_bounds[0])]
    ub_parts = [np.full(3, cfg.ua_adj_bounds[1])]

    if cfg.free_ua_loss_bottom:
        x0_parts.append(np.array([float(ua_loss_frozen[0])]))
        lb_parts.append(np.array([-0.008]))
        ub_parts.append(np.array([0.004]))

    if cfg.free_f_ashp:
        # Initial guess: split heat equally between mid-hi and top, none to mid
        x0_parts.append(np.array([0.0, 0.25]))
        lb_parts.append(np.full(2, cfg.f_ashp_mid_bounds[0]))
        ub_parts.append(np.full(2, cfg.f_ashp_mid_bounds[1]))

    x0 = np.concatenate(x0_parts)
    lb = np.concatenate(lb_parts)
    ub = np.concatenate(ub_parts)

    # Clamp x0 within bounds
    x0 = np.clip(x0, lb + 1e-8, ub - 1e-8)

    cost_history: list[float] = []

    # Draw mask: flag steps k where T_bottom drops sharply at k+1.
    # dT_bottom[k] = T_meas[k+1, 0] - T_meas[k, 0]; exclude if < draw_delta_c.
    dT_bottom = np.diff(T_meas[:, 0])          # shape (steps,)
    draw_mask = dT_bottom < cfg.draw_delta_c   # True = draw event, exclude
    n_draw = int(draw_mask.sum())
    logger.info(
        "Draw mask: %d / %d steps excluded (draw_delta_c=%.1f degC)",
        n_draw, steps, cfg.draw_delta_c,
    )

    def residuals(x):
        p = _build_tank_params(x, ua_loss_frozen, cfg.free_ua_loss_bottom, cfg.free_f_ashp)
        errs = np.zeros((steps, 4))
        for k in range(steps):
            T_pred = tank_step(
                T_meas[k],
                float(Q_st[k]),
                float(Q_ashp[k]),
                float(Q_imm[k]),
                float(T_amb[k]),
                p,
            )
            errs[k] = T_pred - T_meas[k + 1]
        # Zero out residuals for draw-contaminated steps so they don't
        # bias UA_adj[0] toward zero.
        errs[draw_mask] = 0.0
        flat = errs.ravel()
        cost_history.append(float(np.sum(flat ** 2)))
        return flat

    result = least_squares(
        residuals,
        x0,
        bounds=(lb, ub),
        method="trf",
        max_nfev=cfg.max_nfev,
        verbose=0,
    )
    logger.info("Global fit cost: %.4f, nfev: %d", result.cost, result.nfev)

    fitted_params = _build_tank_params(
        result.x, ua_loss_frozen, cfg.free_ua_loss_bottom, cfg.free_f_ashp
    )

    # Compute per-node train RMSE
    T_pred_all = np.zeros((steps, 4))
    for k in range(steps):
        T_pred_all[k] = tank_step(
            T_meas[k],
            float(Q_st[k]),
            float(Q_ashp[k]),
            float(Q_imm[k]),
            float(T_amb[k]),
            fitted_params,
        )
    node_names = ["T_bottom", "T_mid", "T_mid_hi", "T_top"]
    train_rmse = {}
    for i, name in enumerate(node_names):
        err = T_pred_all[:, i] - T_meas[1:, i]
        train_rmse[name] = float(np.sqrt(np.mean(err ** 2)))

    return GlobalFitResult(
        tank_params=fitted_params,
        cost_history=cost_history,
        train_rmse=train_rmse,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_global_fit(cfg: Optional[GlobalFitConfig] = None) -> GlobalFitResult:
    """Run the full global fitting pipeline.

    Parameters
    ----------
    cfg : GlobalFitConfig, optional
        If None, uses defaults.

    Returns
    -------
    GlobalFitResult with fitted TankParams, cost history, and per-node RMSE.
    """
    if cfg is None:
        cfg = GlobalFitConfig()

    # Load priors
    ua_loss = _load_ua_priors(cfg.ua_fit_path)
    ashp_params = _load_ashp_priors(cfg.ashp_fit_path)

    # Load data
    logger.info("Loading data from %s ...", cfg.data_csv)
    df = load_and_clean(cfg.data_csv, cfg.column_mapping_yaml)

    # Compute ST energy
    df["st_kwh"] = compute_st_energy(df)

    # Drop rows where all tank temperatures are NaN
    tank_cols = ["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]
    df = df.dropna(subset=tank_cols, how="all")

    # Train/val split
    split_idx = int(len(df) * cfg.train_frac)
    df_train = df.iloc[:split_idx]
    logger.info("Train: %d rows, Val: %d rows", split_idx, len(df) - split_idx)

    # Prepare inputs
    inputs = _prepare_inputs(df_train, ashp_params)

    # Fit
    result = _fit(inputs, ua_loss, cfg)
    logger.info("Fitted UA_adj: %s", result.tank_params.UA_adj.tolist())
    logger.info("Final UA_loss: %s", result.tank_params.UA_loss.tolist())
    logger.info("Train RMSE: %s", result.train_rmse)

    return result
