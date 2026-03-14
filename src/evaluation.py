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
    Q_back: pd.Series = None
) -> dict[str, float]:
    """
    Compute COP or power-prediction error statistics.

    If Q_back is provided, computes COP error metrics between model and measured COP (Q_back / ashp_inst_kwh).
    Otherwise, computes power error metrics as before.
    Returns dict with keys: median_ape, mean_ape, rmse, n_samples.
    """
    if Q_back is not None:
        # COP error: only intervals with valid Q_back and ashp_inst_kwh > 0.05
        idx = Q_back.notna() & (df["ashp_inst_kwh"].fillna(0) > 0.05)
        if idx.sum() < 10:
            return {"median_ape": np.nan, "mean_ape": np.nan, "rmse": np.nan, "n_samples": int(idx.sum())}
        sub = df.loc[idx]
        Qb = Q_back[idx].values
        P_meas = sub["ashp_inst_kwh"].values
        # Avoid division by zero
        COP_meas = np.clip(Qb / np.maximum(P_meas, 0.01), 0.1, 8.0)
        T_sink = ashp_model.sink_proxy(sub["tank_mid_c"].values, sub["tank_top_c"].values)
        COP_pred = np.clip(ashp_model.predict_cop(sub["t_out_c"].values, T_sink, ashp_p), 0.1, 8.0)
        ape = np.abs(COP_pred - COP_meas) / np.maximum(COP_meas, 0.1) * 100.0
        rmse = float(np.sqrt(np.nanmean((COP_pred - COP_meas) ** 2)))
        return {
            "median_ape": float(np.nanmedian(ape)),
            "mean_ape": float(np.nanmean(ape)),
            "rmse": rmse,
            "n_samples": int(idx.sum()),
        }
    else:
        # Power error (legacy behavior)
        p_all = df["ashp_inst_kwh"].fillna(0)
        valid = p_all > 0.05
        if valid.sum() < 20:
            return {"median_ape": np.nan, "mean_ape": np.nan, "rmse": np.nan, "n_samples": int(valid.sum())}
        p75 = p_all[valid].quantile(0.75)
        mask = p_all >= p75
        if mask.sum() < 10:
            mask = valid
        sub = df.loc[mask]
        T_sink = ashp_model.sink_proxy(sub["tank_mid_c"].values, sub["tank_top_c"].values)
        P_pred = ashp_model.predict_power(sub["t_out_c"].values, T_sink, ashp_p)
        P_pred_kwh = P_pred * dt_h
        P_meas = sub["ashp_inst_kwh"].values
        ape = np.abs(P_pred_kwh - P_meas) / np.maximum(P_meas, 0.05) * 100.0
        return {
            "median_ape": float(np.nanmedian(ape)),
            "mean_ape": float(np.nanmean(ape)),
            "rmse": float(np.sqrt(np.nanmean((P_pred_kwh - P_meas) ** 2))),
            "n_samples": int(mask.sum()),
        }


def ashp_performance_kpis(
    df: pd.DataFrame,
    ashp_p: ashp_model.ASHPParams,
    dt_h: float = 0.5,
) -> dict[str, float]:
    """Compute ASHP performance KPIs over the evaluation period.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with ASHP, tank, and temperature columns.
    ashp_p : ashp_model.ASHPParams
        Fitted ASHP parameters.
    dt_h : float
        Interval length in hours (default 0.5).

    Returns
    -------
    dict[str, float]
        Keys: spf, mean_cop_on, frac_cop_above_3, ashp_runtime_frac.
    """
    nan_result = {
        "spf": float("nan"),
        "mean_cop_on": float("nan"),
        "frac_cop_above_3": float("nan"),
        "ashp_runtime_frac": float("nan"),
    }

    ashp_on = df["ashp_inst_kwh"].fillna(0) > 0.05
    n_on = ashp_on.sum()
    ashp_runtime_frac = float(n_on) / len(df) if len(df) > 0 else 0.0

    if n_on < 10:
        return nan_result

    sub = df.loc[ashp_on]
    T_sink = ashp_model.sink_proxy(sub["tank_mid_c"].values, sub["tank_top_c"].values)
    cop = ashp_model.predict_cop(sub["t_out_c"].values, T_sink, ashp_p)

    # Clip COP to plausible range
    cop = np.clip(cop, 0.5, 8.0)

    P_meas = sub["ashp_inst_kwh"].values
    Q_pred = P_meas * cop  # predicted heat [kWh]

    spf = float(np.nansum(Q_pred) / np.nansum(P_meas))
    mean_cop_on = float(np.nanmean(cop))
    frac_cop_above_3 = float(np.nanmean(cop >= 3.0))

    return {
        "spf": spf,
        "mean_cop_on": mean_cop_on,
        "frac_cop_above_3": frac_cop_above_3,
        "ashp_runtime_frac": ashp_runtime_frac,
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


def _one_step_ahead(
    T_meas: np.ndarray,
    Q_st: np.ndarray,
    Q_ashp: np.ndarray,
    Q_imm: np.ndarray,
    T_amb: np.ndarray,
    params: tank_model.TankParams,
) -> np.ndarray:
    """One-step-ahead prediction: each step resets to the measured state.

    Returns T_pred of shape (N-1, 4) — predicted temperatures at steps 1..N-1.
    """
    N = len(T_meas)
    T_pred = np.zeros((N - 1, 4))
    for k in range(N - 1):
        T_pred[k] = tank_model.tank_step(
            T_meas[k],
            float(Q_st[k]), float(Q_ashp[k]),
            float(Q_imm[k]), float(T_amb[k]),
            params,
        )
    return T_pred


def evaluate(

    df: pd.DataFrame,
    id_result: identification.IdentificationResult,
    label: str = "validation",
    plot_dir: Path | None = None,
    Q_back: pd.Series = None,
) -> dict:
    """
    Run the full evaluation on a DataFrame slice.

    Uses one-step-ahead prediction (teacher forcing) for RMSE, which is
    the standard metric for grey-box thermal models.
    If Q_back is not provided, it is computed using back_calculate_ashp_heat.
    Returns a summary dict and optionally saves plots.
    """
    inputs = identification.prepare_inputs(df, id_result.ashp_params)
    T_meas = inputs["T_meas"]

    # One-step-ahead prediction
    T_sim = _one_step_ahead(
        T_meas,
        inputs["Q_st"],
        inputs["Q_ashp"],
        inputs["Q_imm"],
        inputs["T_amb"],
        id_result.tank_params,
    )

    if Q_back is None:
        Q_back = identification.back_calculate_ashp_heat(df)

    rmses = node_rmses(T_meas[1:], T_sim)
    cop_err = cop_errors(df, id_result.ashp_params, Q_back=Q_back)
    ashp_kpis = ashp_performance_kpis(df, id_result.ashp_params)
    ordering = node_ordering_rate(T_sim)
    e_resid = energy_balance_residual(
        T_meas, inputs["Q_st"], inputs["Q_ashp"],
        inputs["Q_imm"], inputs["T_amb"], id_result.tank_params,
    )

    summary = {
        "label": label,
        "node_rmse": rmses,
        "cop_errors": cop_err,
        "ashp_kpis": ashp_kpis,
        "ordering_rate": ordering,
        "energy_balance_residual_kwh": e_resid,
        "n_samples": len(df),
    }

    logger.info("=== %s evaluation ===", label.upper())
    for name, val in rmses.items():
        logger.info("  RMSE %s: %.2f °C", name, val)
    logger.info("  COP median APE: %.1f %%", cop_err.get("median_ape", float("nan")))
    logger.info("  SPF: %.2f", ashp_kpis.get("spf", float("nan")))
    logger.info("  Fraction COP >= 3: %.1f %%", ashp_kpis.get("frac_cop_above_3", float("nan")) * 100)
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
