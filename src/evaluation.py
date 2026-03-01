"""
Evaluation and reporting for the Stage-1 digital twin.

Computes RMSE per node, COP error statistics, energy-balance residuals,
node-ordering violation rate, and generates diagnostic plots.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional — plots only generated when matplotlib is available
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:  # pragma: no cover
    HAS_MPL = False

from . import ashp_model, tank_model, identification

NODE_NAMES = ["T_bottom", "T_mid", "T_mid_hi", "T_top"]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-mean-square error, ignoring NaNs."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def node_rmses(T_meas: np.ndarray, T_sim: np.ndarray) -> dict[str, float]:
    """Compute per-node RMSE.  Both arrays shape (N, 4)."""
    return {NODE_NAMES[i]: rmse(T_meas[:, i], T_sim[:, i]) for i in range(4)}


def cop_errors(
    df: pd.DataFrame,
    ashp_p: ashp_model.ASHPParams,
    dt_h: float = 0.5,
) -> dict[str, float]:
    """Compute COP prediction-error statistics.

    Returns dict with keys: median_ape, mean_ape, rmse.
    """
    mask = df["ashp_inst_kwh"].fillna(0) > 0.02
    if mask.sum() < 10:
        return {"median_ape": np.nan, "mean_ape": np.nan, "rmse": np.nan}

    sub = df.loc[mask]
    T_sink = ashp_model.sink_proxy(sub["tank_mid_c"].values, sub["tank_top_c"].values)
    P_pred = ashp_model.predict_power(sub["t_amb_c"].values, T_sink, ashp_p)
    P_pred_kwh = P_pred * dt_h

    P_meas = sub["ashp_inst_kwh"].values
    ape = np.abs(P_pred_kwh - P_meas) / np.maximum(P_meas, 0.01) * 100.0

    return {
        "median_ape": float(np.median(ape)),
        "mean_ape":   float(np.mean(ape)),
        "rmse":       float(np.sqrt(np.mean((P_pred_kwh - P_meas) ** 2))),
    }


def node_ordering_rate(T_sim: np.ndarray) -> float:
    """Fraction of steps where T_top >= T_mh >= T_mid >= T_bot (with 0.5 K tolerance)."""
    tol = 0.5
    ordered = (
        (T_sim[:, 3] >= T_sim[:, 2] - tol)
        & (T_sim[:, 2] >= T_sim[:, 1] - tol)
        & (T_sim[:, 1] >= T_sim[:, 0] - tol)
    )
    return float(ordered.mean())


def energy_balance_residual(
    T_meas: np.ndarray,
    Q_st: np.ndarray,
    Q_ashp: np.ndarray,
    Q_imm: np.ndarray,
    T_amb: np.ndarray,
    params: tank_model.TankParams,
) -> float:
    """Return total energy-balance residual [kWh] over the period.

    residual = ΣQ_in - ΣQ_stored - ΣQ_loss  (should be ≈0).
    """
    # Stored energy change
    dT = T_meas[-1] - T_meas[0]  # shape (4,)
    E_stored = np.sum(dT) * tank_model.NODE_CAP / 3600.0  # kJ→kWh

    E_in = np.nansum(Q_st) + np.nansum(Q_ashp) + np.nansum(Q_imm)

    # Approximate total losses
    dt_s = 1800.0
    E_loss = 0.0
    for i in range(4):
        avg_dT = np.nanmean(T_meas[:, i] - T_amb)
        E_loss += params.UA_loss[i] * avg_dT * dt_s * len(T_amb) / 3600.0

    residual = E_in - E_stored - E_loss
    return float(residual)


def evaluate(
    df: pd.DataFrame,
    id_result: identification.IdentificationResult,
    label: str = "validation",
    plot_dir: Path | None = None,
) -> dict:
    """Run the full evaluation on a DataFrame slice.

    Returns a summary dict and optionally saves plots.
    """
    inputs = identification.prepare_inputs(df, id_result.ashp_params)
    T_meas = inputs["T_meas"]
    N = len(T_meas)
    steps = N - 1
    T0 = T_meas[0]
    T_sim_full = tank_model.simulate(
        T0,
        inputs["Q_st"][:steps],
        inputs["Q_ashp"][:steps],
        inputs["Q_imm"][:steps],
        inputs["T_amb"][:steps],
        id_result.tank_params,
    )
    T_sim = T_sim_full[1:]  # shape (steps, 4) — matches T_meas[1:N]

    rmses = node_rmses(T_meas[1:], T_sim)
    cop_err = cop_errors(df, id_result.ashp_params)
    ordering = node_ordering_rate(T_sim)
    e_resid = energy_balance_residual(
        T_meas, inputs["Q_st"], inputs["Q_ashp"],
        inputs["Q_imm"], inputs["T_amb"], id_result.tank_params,
    )

    summary = {
        "label": label,
        "node_rmse": rmses,
        "cop_errors": cop_err,
        "ordering_rate": ordering,
        "energy_balance_residual_kwh": e_resid,
        "n_samples": len(df),
    }

    logger.info("=== %s evaluation ===", label.upper())
    for name, val in rmses.items():
        logger.info("  RMSE %s: %.2f °C", name, val)
    logger.info("  COP median APE: %.1f %%", cop_err.get("median_ape", float("nan")))
    logger.info("  Ordering rate: %.1f %%", ordering * 100)
    logger.info("  Energy balance residual: %.1f kWh", e_resid)

    if plot_dir is not None and HAS_MPL:
        plot_dir = Path(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        _plot_nodes(df.index[1:], T_meas[1:], T_sim, label, plot_dir)

    return summary


def _plot_nodes(
    time_idx: pd.DatetimeIndex,
    T_meas: np.ndarray,
    T_sim: np.ndarray,
    label: str,
    plot_dir: Path,
) -> None:
    """Generate per-node measured-vs-simulated plots."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for i, (ax, name) in enumerate(zip(axes, NODE_NAMES)):
        ax.plot(time_idx, T_meas[:, i], "k-", lw=0.8, label="Measured")
        ax.plot(time_idx, T_sim[:, i], "r--", lw=0.8, label="Simulated")
        ax.set_ylabel(f"{name} [°C]")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_title(f"Tank node temperatures — {label}")
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(plot_dir / f"nodes_{label}.png", dpi=150)
    plt.close(fig)
    logger.info("Saved node plot to %s", plot_dir / f"nodes_{label}.png")
