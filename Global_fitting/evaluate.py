"""
Global_fitting.evaluate – Evaluation and diagnostic plotting for the global
fitting pipeline.

Computes per-node RMSE and MAE on train and validation splits using
one-step-ahead prediction (consistent with how parameters were fitted).

Produces 4-panel time-series plots of one-step-ahead prediction error for
each node — one figure for train, one for validation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.ashp_model import ASHPParams, predict_cop, sink_proxy
from src.solar_thermal import compute_st_energy
from src.tank_model import NODE_CAP, TankParams, tank_step

logger = logging.getLogger(__name__)

# Optional matplotlib
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MPL = True
except ImportError:  # pragma: no cover
    HAS_MPL = False

NODE_NAMES = ["T_bottom", "T_mid", "T_mid_hi", "T_top"]
_NODE_LABELS = ["Bottom", "Mid", "Mid-Hi", "Top"]
_NODE_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def _prepare_inputs(
    df,
    ashp_params: ASHPParams,
    ua_loss: np.ndarray | None = None,
    dt_h: float | None = None,
    sampling_minutes: int = 5,
    ua_adj: np.ndarray | None = None,
):
    """Build arrays needed for one-step-ahead prediction.

    Q_ashp is back-calculated from the measured energy balance on nodes 1–3,
    gated by a minimal mask (ASHP on, ST off, immersion off).  This avoids
    dependence on the COP map or the restrictive top_rising detection gate.

    Parameters
    ----------
    ua_loss : array (4,)
        Per-node UA to ambient [kW/K].  Required for the back-calculation.
    dt_h : float, optional
        Time-step in hours.  Derived from *sampling_minutes* if not given.
    sampling_minutes : int
        Interval cadence [minutes].  Used when *dt_h* is None.
    ua_adj : array (3,), optional
        Inter-node conductances [kW/K].  When provided, the heat conducted
        from mid (node 1) to bottom (node 0) via ua_adj[0] is added back to
        Q_ashp to account for heat leaving the 3-node accounting boundary.
    """
    if dt_h is None:
        dt_h = sampling_minutes / 60.0
    if ua_loss is None:
        raise ValueError("ua_loss is required for back-calculated Q_ashp")

    T_meas = df[["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]].values

    if "st_kwh" in df.columns:
        Q_st = df["st_kwh"].fillna(0).values
    else:
        Q_st = compute_st_energy(df, dt_minutes=dt_h * 60).values

    P_meas = df["ashp_inst_kwh"].fillna(0).values
    Q_imm  = df["imm_tot_inst_kwh"].fillna(0).values
    T_amb  = df["t_amb_c"].fillna(df["t_amb_c"].median()).values

    ashp_on = P_meas > 0.016
    st_off  = Q_st <= 0.001
    imm_off = Q_imm <= 0.01
    ashp_dhw_gate = ashp_on & st_off & imm_off

    dt_s = dt_h * 3600.0
    N = len(P_meas)
    Q_ashp = np.zeros(N)
    for k in range(1, N):
        if not ashp_dhw_gate[k]:
            continue
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
        Q_ashp[k] = max(Q_kJ / 3600.0, 0.0)

    return dict(T_meas=T_meas, Q_st=Q_st, Q_ashp=Q_ashp, Q_imm=Q_imm, T_amb=T_amb)


def _one_step_ahead(inputs: dict, params: TankParams,
                    dt_s: float = 300.0) -> np.ndarray:
    """One-step-ahead prediction: each step resets to the measured state.

    Returns T_pred of shape (N-1, 4).
    """
    T_meas = inputs["T_meas"]
    Q_st = inputs["Q_st"]
    Q_ashp = inputs["Q_ashp"]
    Q_imm = inputs["Q_imm"]
    T_amb = inputs["T_amb"]
    N = len(Q_st)
    T_pred = np.zeros((N - 1, 4))
    for k in range(N - 1):
        T_pred[k] = tank_step(
            T_meas[k],
            float(Q_st[k]),
            float(Q_ashp[k]),
            float(Q_imm[k]),
            float(T_amb[k]),
            params,
            dt_s,
        )
    return T_pred


def evaluate_split(
    df,
    params: TankParams,
    ashp_params: ASHPParams,
    label: str = "train",
    ua_loss: np.ndarray | None = None,
    sampling_minutes: int = 5,
    ua_adj: np.ndarray | None = None,
) -> dict:
    """Compute per-node RMSE and MAE using one-step-ahead prediction.

    Parameters
    ----------
    df : pd.DataFrame
        Data slice (train or validation).
    params : TankParams
        Fitted tank parameters.
    ashp_params : ASHPParams
        Frozen ASHP parameters (retained for API compatibility).
    label : str
        Label for logging.
    ua_loss : array (4,)
        Per-node UA to ambient [kW/K].  Falls back to params.UA_loss.
    sampling_minutes : int
        Interval cadence [minutes].
    ua_adj : array (3,), optional
        Inter-node conductances [kW/K].  Passed to _prepare_inputs for the
        bottom-boundary correction on back-calculated Q_ashp.

    Returns
    -------
    dict with keys: label, node_rmse, node_mae, n_intervals.
    """
    if ua_loss is None:
        ua_loss = params.UA_loss
    dt_s = sampling_minutes * 60.0
    inputs = _prepare_inputs(df, ashp_params, ua_loss=ua_loss,
                             sampling_minutes=sampling_minutes, ua_adj=ua_adj)
    T_meas = inputs["T_meas"]
    T_pred = _one_step_ahead(inputs, params, dt_s=dt_s)

    errors = T_pred - T_meas[1:]
    node_rmse = {}
    node_mae = {}
    for i, name in enumerate(NODE_NAMES):
        node_rmse[name] = float(np.sqrt(np.mean(errors[:, i] ** 2)))
        node_mae[name] = float(np.mean(np.abs(errors[:, i])))

    logger.info("=== %s evaluation ===", label.upper())
    for name in NODE_NAMES:
        logger.info("  RMSE %s: %.3f °C  |  MAE: %.3f °C",
                     name, node_rmse[name], node_mae[name])

    return {
        "label": label,
        "node_rmse": node_rmse,
        "node_mae": node_mae,
        "n_intervals": len(T_pred),
    }


def plot_prediction_errors(
    df,
    params: TankParams,
    ashp_params: ASHPParams,
    label: str = "train",
    output_dir: Optional[Path] = None,
    ua_loss: np.ndarray | None = None,
    sampling_minutes: int = 5,
    ua_adj: np.ndarray | None = None,
) -> Optional[Path]:
    """4-panel time-series plot of one-step-ahead prediction error per node.

    Parameters
    ----------
    df : pd.DataFrame
    params : TankParams
    ashp_params : ASHPParams
    label : str
    output_dir : Path, optional
    ua_loss : array (4,)
        Per-node UA to ambient [kW/K].  Falls back to params.UA_loss.
    sampling_minutes : int
        Interval cadence [minutes].
    ua_adj : array (3,), optional
        Inter-node conductances [kW/K].  Passed to _prepare_inputs for the
        bottom-boundary correction on back-calculated Q_ashp.

    Returns
    -------
    Path to saved figure, or None if matplotlib is unavailable.
    """
    if not HAS_MPL:
        logger.warning("matplotlib not available; skipping plots.")
        return None

    if ua_loss is None:
        ua_loss = params.UA_loss
    dt_s = sampling_minutes * 60.0
    inputs = _prepare_inputs(df, ashp_params, ua_loss=ua_loss,
                             sampling_minutes=sampling_minutes, ua_adj=ua_adj)
    T_meas = inputs["T_meas"]
    T_pred = _one_step_ahead(inputs, params, dt_s=dt_s)
    errors = T_pred - T_meas[1:]

    time_idx = df.index[1:]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"One-step-ahead prediction error — {label}", fontsize=13)

    for i, ax in enumerate(axes):
        ax.plot(
            time_idx, errors[:, i],
            "-", color=_NODE_COLOURS[i], linewidth=0.6, alpha=0.8,
        )
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        rmse_val = float(np.sqrt(np.mean(errors[:, i] ** 2)))
        ax.set_ylabel(f"{_NODE_LABELS[i]} error [°C]")
        ax.legend([f"RMSE = {rmse_val:.3f} °C"], loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()

    saved_path = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = output_dir / f"prediction_error_{label}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        saved_path = fname
        logger.info("Saved prediction error plot: %s", fname)

    plt.close(fig)
    return saved_path
