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

from src.ashp_model import ASHPParams, predict_capacity, predict_cop, sink_proxy
from src.solar_thermal import compute_st_energy
from src.tank_model import TankParams, tank_step

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


def _prepare_inputs(df, ashp_params: ASHPParams, dt_h: float = 0.5):
    """Build arrays needed for one-step-ahead prediction."""
    T_meas = df[["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]].values

    if "st_kwh" in df.columns:
        Q_st = df["st_kwh"].fillna(0).values
    else:
        Q_st = compute_st_energy(df, dt_minutes=dt_h * 60).values

    T_sink = sink_proxy(df["tank_mid_c"].values, df["tank_top_c"].values)
    T_out = df["t_out_c"].fillna(df["t_out_c"].median()).values
    cap_kw = predict_capacity(T_out, T_sink, ashp_params)
    cop    = predict_cop(T_out, T_sink, ashp_params)
    P_meas = df["ashp_inst_kwh"].fillna(0).values
    top_rising = pd.Series(df["tank_top_c"].values, index=df.index).diff().fillna(0.0).values > 1.0
    ashp_dhw_on = (P_meas > 0.013) & top_rising
    Q_ashp = np.where(ashp_dhw_on, P_meas * cop, 0.0)
    Q_ashp = np.concatenate([Q_ashp[1:], [0.0]])

    Q_imm = df["imm_tot_inst_kwh"].fillna(0).values
    T_amb = df["t_amb_c"].fillna(df["t_amb_c"].median()).values

    return dict(T_meas=T_meas, Q_st=Q_st, Q_ashp=Q_ashp, Q_imm=Q_imm, T_amb=T_amb)


def _one_step_ahead(inputs: dict, params: TankParams) -> np.ndarray:
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
        )
    return T_pred


def evaluate_split(
    df,
    params: TankParams,
    ashp_params: ASHPParams,
    label: str = "train",
) -> dict:
    """Compute per-node RMSE and MAE using one-step-ahead prediction.

    Parameters
    ----------
    df : pd.DataFrame
        Data slice (train or validation).
    params : TankParams
        Fitted tank parameters.
    ashp_params : ASHPParams
        Frozen ASHP parameters.
    label : str
        Label for logging.

    Returns
    -------
    dict with keys: label, node_rmse, node_mae, n_intervals.
    """
    inputs = _prepare_inputs(df, ashp_params)
    T_meas = inputs["T_meas"]
    T_pred = _one_step_ahead(inputs, params)

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
) -> Optional[Path]:
    """4-panel time-series plot of one-step-ahead prediction error per node.

    Parameters
    ----------
    df : pd.DataFrame
    params : TankParams
    ashp_params : ASHPParams
    label : str
    output_dir : Path, optional

    Returns
    -------
    Path to saved figure, or None if matplotlib is unavailable.
    """
    if not HAS_MPL:
        logger.warning("matplotlib not available; skipping plots.")
        return None

    inputs = _prepare_inputs(df, ashp_params)
    T_meas = inputs["T_meas"]
    T_pred = _one_step_ahead(inputs, params)
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
