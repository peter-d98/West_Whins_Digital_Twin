"""
Global_fitting.fit_global – Fit free tank parameters (UA_adj and optionally
UA_loss[0]) using frozen priors from UA_fitting and ASHP_fitting.

The loss function uses **one-step-ahead prediction**: at each training
interval *k*, the measured temperatures T[k−1] are taken as the initial
state, ``tank_step()`` is called once to predict T_pred[k], and the residual
is T_pred[k] − T_meas[k].  This resets the model state from measurements at
every step, preventing drift accumulation from unmodelled effects (draws,
partial-interval ASHP runs, etc.).

Physical constraints baked in as hard-coded fixed values:

- ``f_st   = [1, 0, 0, 0]`` — ST coil is physically inside the bottom node
- ``f_ashp``                — loaded from ``ashp_fit.json`` (empirical median
  computed from window ΔT).  Typical: [0, 0.61, 0.25, 0.14].
- ``f_imm  = [0, 1, 0, 0]`` — immersion element is in the mid node

Two categories of steps are excluded from the OSA residual:
- **Draw steps**     — bottom-node temperature drops sharply (hot water drawn).
- **Collapse steps** — mid-node temperature rises sharply with no ASHP power
  draw (HX pump starts, causing stratification redistribution the model does
  not represent); would otherwise bias UA_adj upward.
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
from src.ashp_model import ASHPParams, predict_cop, sink_proxy
from src.data_loader import load_and_clean
from src.solar_thermal import compute_st_energy
from src.tank_model import NODE_CAP, TankParams, tank_step

logger = logging.getLogger(__name__)

# Hard-coded physical heat-distribution fractions (fallbacks only)
F_ST   = np.array([1.0, 0.0, 0.0, 0.0])
F_ASHP = np.array([0.0, 0.0, 0.0, 1.0])  # overridden by empirical value from ashp_fit.json
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

def _load_ua_priors(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load UA_loss and UA_adj arrays from ua_fit.json.

    Returns
    -------
    ua_loss : np.ndarray, shape (4,)
    ua_adj  : np.ndarray, shape (3,)
    """
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
    ua_adj = np.array(data.get("UA_adj", [0.05, 0.05, 0.05]), dtype=float)
    if ua_adj.shape != (3,):
        raise ValueError(
            f"Expected UA_adj of shape (3,), got {ua_adj.shape} from {path}"
        )
    logger.info("Loaded UA_loss priors: %s", ua_loss.tolist())
    logger.info("Loaded UA_adj priors (warm-start): %s", ua_adj.tolist())
    return ua_loss, ua_adj


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
        c=np.array(ashp["c"], dtype=float) if "c" in ashp else None,
        a=np.array(ashp["a"], dtype=float) if "a" in ashp else ASHPParams().a,
        b=np.array(ashp["b"], dtype=float) if "b" in ashp else ASHPParams().b,
    )
    logger.info("Loaded ASHP priors: c=%s",
                params.c.tolist() if params.c is not None else None)
    return params


def _load_f_ashp_prior(path: Path) -> np.ndarray:
    """Load the empirical f_ashp distribution from ashp_fit.json.

    Falls back to the hard-coded ``F_ASHP`` default if the file or key is
    absent (e.g. the ASHP fitting pipeline has not been re-run after adding
    the f_ashp computation step).
    """
    if not path.exists():
        logger.warning(
            "ashp_fit.json not found at %s; using default F_ASHP=%s",
            path, F_ASHP.tolist(),
        )
        return F_ASHP.copy()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    f = data.get("ashp", {}).get("f_ashp")
    if f is None:
        logger.warning(
            "f_ashp key absent in %s (re-run ASHP fitting); using default F_ASHP=%s",
            path, F_ASHP.tolist(),
        )
        return F_ASHP.copy()
    arr = np.array(f, dtype=float)
    if arr.shape != (4,):
        raise ValueError(f"Expected f_ashp of shape (4,), got {arr.shape} from {path}")
    logger.info("Loaded empirical f_ashp: %s", arr.tolist())
    return arr


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def _prepare_inputs(
    df,
    ua_loss: np.ndarray,
    dt_h: float | None = None,
    sampling_minutes: int = 5,
    ua_adj: np.ndarray | None = None,
):
    """Build arrays needed for one-step-ahead prediction.

    Q_ashp is back-calculated from the measured energy balance on nodes 1–3,
    gated by a minimal mask (ASHP on, ST off, immersion off).

    Parameters
    ----------
    ua_loss : array (4,)
        Per-node UA to ambient [kW/K].  Required for the back-calculation.
    dt_h : float, optional
        Time-step in hours.  Derived from *sampling_minutes* if not given.
    sampling_minutes : int
        Interval cadence [minutes].  Used when *dt_h* is None.
    ua_adj : array (3,), optional
        Inter-node conductances [kW/K] from ua_fit.json.  When provided, the
        heat conducted from mid (node 1) to bottom (node 0) via ua_adj[0] is
        added back to Q_ashp so that the back-calculation accounts for heat
        leaving the 3-node accounting boundary.

    Returns dict with keys: T_meas (N,4), Q_st, Q_ashp, Q_imm, T_amb (all N,).
    """
    if dt_h is None:
        dt_h = sampling_minutes / 60.0
    T_meas = df[["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]].values

    # Solar-thermal energy
    if "st_kwh" in df.columns:
        Q_st = df["st_kwh"].fillna(0).values
    else:
        Q_st = compute_st_energy(df, dt_minutes=dt_h * 60).values

    # ASHP heat: back-calculated from the measured energy balance on
    # nodes 1–3 (mid, mid-hi, top) — the nodes directly charged by the
    # ASHP HX circuit.  Bottom is excluded because water is drawn from
    # mid and returned to top; bottom only warms via inter-node conduction.
    #
    # Minimal gate: ASHP drawing power AND solar-thermal / immersion off.
    # During space-heating-only intervals (tank receives no ASHP heat) the
    # energy balance naturally returns ≈ 0 or slightly negative (clipped).
    P_meas = df["ashp_inst_kwh"].fillna(0).values
    Q_imm  = df["imm_tot_inst_kwh"].fillna(0).values
    if "st_kwh" in df.columns:
        Q_st_raw = df["st_kwh"].fillna(0).values
    else:
        Q_st_raw = Q_st  # already computed above

    ashp_on = P_meas > 0.07
    st_off  = Q_st_raw <= 0.001
    imm_off = Q_imm <= 0.01
    ashp_dhw_gate = ashp_on & st_off & imm_off

    T_amb = df["t_amb_c"].fillna(df["t_amb_c"].median()).values
    dt_s = dt_h * 3600.0

    N = len(P_meas)
    Q_ashp = np.zeros(N)
    for k in range(1, N):
        if not ashp_dhw_gate[k]:
            continue
        # Energy balance on nodes 1–3 only
        storage_kJ = sum(
            NODE_CAP[i] * (T_meas[k, i] - T_meas[k - 1, i])
            for i in range(1, 4)
        )
        loss_kJ = sum(
            ua_loss[i] * (T_meas[k - 1, i] - T_amb[k]) * dt_s
            for i in range(1, 4)
        )
        Q_kJ = storage_kJ + loss_kJ
        # Boundary correction: heat conducted from mid (node 1) to bottom
        # (node 0) via ua_adj[0] leaves the 3-node accounting boundary.
        # Guard: only apply when Q_kJ > 0 (i.e. genuine DHW charging), not
        # during SH-only intervals where T_mid > T_bot from prior stratification
        # would otherwise introduce a false positive Q_ashp contribution.
        if ua_adj is not None and Q_kJ > 0:
            Q_kJ += float(ua_adj[0]) * (T_meas[k - 1, 1] - T_meas[k - 1, 0]) * dt_s
        Q_ashp[k] = max(Q_kJ / 3600.0, 0.0)  # clip negative to zero

    return dict(T_meas=T_meas, Q_st=Q_st, Q_ashp=Q_ashp, Q_imm=Q_imm, T_amb=T_amb)


# ---------------------------------------------------------------------------
# Core fitting
# ---------------------------------------------------------------------------

def _build_tank_params(
    x: np.ndarray,
    ua_loss_frozen: np.ndarray,
    free_ua_loss_bottom: bool,
    f_ashp_prior: np.ndarray,
) -> TankParams:
    """Reconstruct a full TankParams from the free-parameter vector *x*.

    Vector layout:
      x[0:3]  = UA_adj (3 values, always free)
      x[3]    = UA_loss[0]  (only if free_ua_loss_bottom=True)
    """
    p = TankParams()
    p.UA_adj = x[0:3]

    ua_loss = ua_loss_frozen.copy()
    if free_ua_loss_bottom:
        ua_loss[0] = x[3]
    p.UA_loss = ua_loss

    p.f_st   = F_ST.copy()
    p.f_imm  = F_IMM.copy()
    p.f_ashp = f_ashp_prior.copy()

    return p


def _fit(
    inputs: dict,
    ua_loss_frozen: np.ndarray,
    cfg: GlobalFitConfig,
    *,
    ua_adj_prior: np.ndarray | None = None,
    f_ashp_prior: np.ndarray | None = None,
) -> GlobalFitResult:
    """Run least_squares optimisation for UA_adj (+ optionally UA_loss[0])."""
    T_meas = inputs["T_meas"]
    Q_st = inputs["Q_st"]
    Q_ashp = inputs["Q_ashp"]
    Q_imm = inputs["Q_imm"]
    T_amb = inputs["T_amb"]
    N = len(Q_st)
    steps = N - 1

    f_ashp = f_ashp_prior if f_ashp_prior is not None else F_ASHP.copy()

    # Build initial guess and bounds.
    # Warm-start UA_adj from idle-fitted values when available; the generic
    # [0.05, 0.05, 0.05] guess is ~7x higher and risks a poor local minimum.
    ua_adj_x0 = ua_adj_prior if ua_adj_prior is not None else np.array([0.05, 0.05, 0.05])
    x0_parts = [ua_adj_x0.copy()]
    lb_parts = [np.full(3, cfg.ua_adj_bounds[0])]
    ub_parts = [np.full(3, cfg.ua_adj_bounds[1])]

    if cfg.free_ua_loss_bottom:
        x0_parts.append(np.array([float(ua_loss_frozen[0])]))
        lb_parts.append(np.array([-0.008]))
        ub_parts.append(np.array([0.004]))

    x0 = np.concatenate(x0_parts)
    lb = np.concatenate(lb_parts)
    ub = np.concatenate(ub_parts)

    # Clamp x0 within bounds
    x0 = np.clip(x0, lb + 1e-8, ub - 1e-8)

    cost_history: list[float] = []
    dt_s = cfg.sampling_minutes * 60.0

    # Draw mask: flag steps k where T_bottom drops sharply at k+1.
    # dT_bottom[k] = T_meas[k+1, 0] - T_meas[k, 0]; exclude if < draw_delta_c.
    dT_bottom = np.diff(T_meas[:, 0])          # shape (steps,)
    draw_mask = dT_bottom < cfg.draw_delta_c
    n_draw = int(draw_mask.sum())
    logger.info(
        "Draw mask: %d / %d steps excluded (draw_delta_c=%.2f °C)",
        n_draw, steps, cfg.draw_delta_c,
    )

    # Collapse mask: flag steps k where T_mid rises sharply at k+1 WITHOUT
    # ASHP power draw.  These are HX-pump stratification-redistribution events
    # that the lumped model cannot represent; including them would bias UA_adj
    # upward to partially explain the apparent mid-node heating.
    dT_mid = np.diff(T_meas[:, 1])             # shape (steps,)
    # Q_ashp[k] is the heat attributed to step k→k+1 (same indexing as errs[k]).
    ashp_idle = Q_ashp[:steps] < cfg.ashp_off_kwh
    collapse_mask = (dT_mid > cfg.collapse_mid_rising_c) & ashp_idle
    n_collapse = int(collapse_mask.sum())
    logger.info(
        "Collapse mask: %d / %d steps excluded (mid_rising_c=%.1f °C, no ASHP power)",
        n_collapse, steps, cfg.collapse_mid_rising_c,
    )

    exclude_mask = draw_mask | collapse_mask

    def residuals(x):
        p = _build_tank_params(x, ua_loss_frozen, cfg.free_ua_loss_bottom, f_ashp)
        errs = np.zeros((steps, 4))
        for k in range(steps):
            T_pred = tank_step(
                T_meas[k],
                float(Q_st[k]),
                float(Q_ashp[k]),
                float(Q_imm[k]),
                float(T_amb[k]),
                p,
                dt_s,
            )
            errs[k] = T_pred - T_meas[k + 1]
        # Zero out residuals for draw and collapse steps.
        errs[exclude_mask] = 0.0
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
        result.x, ua_loss_frozen, cfg.free_ua_loss_bottom, f_ashp
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
            dt_s,
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
    ua_loss, ua_adj = _load_ua_priors(cfg.ua_fit_path)
    f_ashp = _load_f_ashp_prior(cfg.ashp_fit_path)

    # Load data
    logger.info("Loading data from %s ...", cfg.data_csv)
    df = load_and_clean(cfg.data_csv, cfg.column_mapping_yaml,
                        sampling_minutes=cfg.sampling_minutes)

    # Compute ST energy
    df["st_kwh"] = compute_st_energy(df, dt_minutes=float(cfg.sampling_minutes))

    # Drop rows where all tank temperatures are NaN
    tank_cols = ["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]
    df = df.dropna(subset=tank_cols, how="all")

    # Train/val split
    split_idx = int(len(df) * cfg.train_frac)
    df_train = df.iloc[:split_idx]
    logger.info("Train: %d rows, Val: %d rows", split_idx, len(df) - split_idx)

    # Prepare inputs
    inputs = _prepare_inputs(df_train, ua_loss,
                             sampling_minutes=cfg.sampling_minutes,
                             ua_adj=ua_adj)

    # Fit
    result = _fit(inputs, ua_loss, cfg, ua_adj_prior=ua_adj, f_ashp_prior=f_ashp)
    logger.info("Fitted UA_adj: %s", result.tank_params.UA_adj.tolist())
    logger.info("Final UA_loss: %s", result.tank_params.UA_loss.tolist())
    logger.info("Train RMSE: %s", result.train_rmse)

    return result
