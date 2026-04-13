"""
ST_fitting.evaluate – Diagnostic plotting for regression-based ST fitting.

Produces:
1. Regression diagnostics (GTI vs Q_meas scatter, T_flow residuals).
2. Free-forward simulation plots for selected ST windows using the fitted
   Q_sol energy map + T_flow bilinear map + dynamic node weights.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from ST_fitting.config import STFitConfig
from ST_fitting.detector import STWindow
from ST_fitting.st_model import (
    predict_q_sol_kwh,
    predict_t_flow_c,
    st_node_weights_from_tflow,
)
from src.tank_model import TankParams, STModelParams, simulate, NODE_CAP

logger = logging.getLogger(__name__)

_NODE_LABELS = ["Bottom", "Mid", "Mid-Hi", "Top"]
_NODE_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


# ---------------------------------------------------------------------------
# Public: write diagnostic CSVs
# ---------------------------------------------------------------------------

def write_interval_csv(
    diag_df: pd.DataFrame,
    *,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        diag_df.to_csv(output_path, index=False)
        logger.info("Interval diagnostics saved to %s", output_path)
    return diag_df


def write_window_csv(
    window_df: pd.DataFrame,
    *,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        window_df.to_csv(output_path, index=False)
        logger.info("Window summary CSV saved to %s", output_path)
    return window_df


# ---------------------------------------------------------------------------
# Public: Regression diagnostics plot
# ---------------------------------------------------------------------------

def plot_regression_diagnostics(
    reg_df: pd.DataFrame,
    result: Dict,
    *,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """Two-panel scatter: GTI vs Q_meas (with fit line), GTI vs T_flow residuals."""
    q = result["Q_sol"]
    t = result["T_flow"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Q_sol scatter
    gti = reg_df["GTI"].values
    q_meas = reg_df["q_meas_kwh"].values
    ax1.scatter(gti, q_meas, s=8, alpha=0.4, label="data")
    gti_line = np.linspace(gti.min(), gti.max(), 100)
    q_line = q["q0_kwh"] + q["q1_kwh_per_wm2"] * gti_line
    ax1.plot(gti_line, q_line, "r-", linewidth=2, label=f"fit (R²={q['r2']:.4f})")
    ax1.set_xlabel("GTI [W/m²]")
    ax1.set_ylabel("Q_meas [kWh]")
    ax1.set_title("Q_sol regression")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: T_flow residuals
    t_bottom = reg_df["t_bottom_c"].values
    t_flow_meas = reg_df["st_flow_temp_c"].values
    t_flow_pred = t["b0_c"] + t["b1"] * t_bottom + t["b2_c_per_wm2"] * gti
    residuals = t_flow_meas - t_flow_pred
    ax2.scatter(gti, residuals, s=8, alpha=0.4, c="tab:orange")
    ax2.axhline(0, color="k", linewidth=0.8)
    ax2.set_xlabel("GTI [W/m²]")
    ax2.set_ylabel("T_flow residual [°C]")
    ax2.set_title(f"T_flow residuals (R²={t['r2']:.4f})")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    saved_path = None
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        saved_path = output_path
        logger.info("Saved regression diagnostics plot: %s", output_path)
    plt.close(fig)
    return saved_path


# ---------------------------------------------------------------------------
# Public: Free-forward window plots
# ---------------------------------------------------------------------------

def plot_freeforward_windows(
    df: pd.DataFrame,
    windows: List[STWindow],
    result: Dict,
    cfg: Optional[STFitConfig] = None,
    *,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """Free-forward simulation for the longest N windows using fitted ST model.

    Uses the dynamic ST pathway: for each interval in the window, predict
    Q_sol and T_flow from GTI → compute dynamic node weights → simulate.
    ST is always active during identified windows (no GTI/T_bottom gate
    applied — the window detector already ensures conditions are met).

    UA parameters are loaded from global_fit.json for losses/conduction.
    """
    if cfg is None:
        cfg = STFitConfig()

    # Load UA params from global fit
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    global_json = _REPO_ROOT / "Global_fitting" / "output" / "global_fit.json"
    if not global_json.exists():
        logger.warning("global_fit.json not found — cannot produce free-forward plots.")
        return []

    with open(global_json, "r") as f:
        gd = json.load(f)

    tank_params = TankParams()
    tank_params.UA_loss = np.array(gd["UA_loss"], dtype=float)
    tank_params.UA_adj = np.array(gd["UA_adj"], dtype=float)
    tank_params.f_ashp = np.array(gd["f_ashp"], dtype=float)
    tank_params.f_imm = np.array(gd["f_imm"], dtype=float)
    tank_params.f_st = np.array(gd.get("f_st", [0.25, 0.25, 0.25, 0.25]), dtype=float)

    st_params = STModelParams.from_dict(result)

    dt_s = cfg.sampling_minutes * 60.0
    dt_h = cfg.sampling_minutes / 60.0
    n_train = int(len(df) * cfg.train_frac)
    df_train = df.iloc[:n_train]

    # Load GTI for the free-forward
    from ST_fitting.fit_st import _load_gti
    _GTI_CSV = _REPO_ROOT / "data" / "hist_GTI_5min.csv"
    gti_series = _load_gti(_GTI_CSV)

    sorted_windows = sorted(windows, key=lambda w: w.n_intervals, reverse=True)
    n_plot = min(cfg.n_plot_windows, len(sorted_windows))
    saved = []

    for w in sorted_windows[:n_plot]:
        first_k = w.indices[0]
        last_k = w.indices[-1]
        seg = df_train.iloc[first_k: last_k + 1].copy()
        N = len(seg)
        if N < 2:
            continue

        node_cols = cfg.node_cols
        T_meas = seg[node_cols].values
        T_amb = seg[cfg.t_amb_col].fillna(seg[cfg.t_amb_col].median()).values

        # Initial condition
        if first_k > 0:
            T0 = df_train[node_cols].iloc[first_k - 1].values.astype(float)
        else:
            T0 = T_meas[0].copy()

        # Build per-interval Q_sol and GTI arrays using the fitted model.
        # ST is always active during a detected window (gates already passed).
        Q_st_pred = np.zeros(N)
        gti_arr = np.zeros(N)
        for i_step in range(N):
            ts = seg.index[i_step]
            gti_val = float(gti_series.get(ts, 0.0))
            gti_arr[i_step] = gti_val
            Q_st_pred[i_step] = predict_q_sol_kwh(
                gti_val, st_params.q0_kwh, st_params.q1_kwh_per_wm2,
            )

        # Simulate with dynamic ST pathway
        Q_ashp = np.zeros(N)
        Q_imm = np.zeros(N)
        T_hist = simulate(
            T0, Q_st_pred, Q_ashp, Q_imm, T_amb, tank_params, dt_s,
            gti=gti_arr, st_params=st_params,
        )
        T_pred = T_hist[1:]

        # Measured Q_st for comparison
        Q_st_meas = (seg[cfg.st_power_col].fillna(0.0) * dt_h).values

        # --- Plot ---
        fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 2, 2, 2, 1.2]})
        fig.suptitle(
            f"Free-forward (fitted ST model)\n"
            f"{seg.index[0]:%Y-%m-%d %H:%M} → {seg.index[-1]:%Y-%m-%d %H:%M}"
            f"  ({N} intervals)",
            fontsize=12,
        )

        T_meas_cmp = T_meas[1:]
        T_pred_cmp = T_pred[:N-1]

        for i in range(4):
            ax = axes[i]
            ax.plot(seg.index, T_meas[:, i], "o-", color=_NODE_COLOURS[i],
                    linewidth=1.4, markersize=3.5, label="Measured", zorder=3)
            ax.plot(seg.index, T_pred[:, i], "--", color=_NODE_COLOURS[i],
                    linewidth=1.4, alpha=0.8, label="Predicted", zorder=3)
            ax.axvspan(seg.index[0], seg.index[-1], alpha=0.10, color="tab:orange", zorder=1)

            if N > 1:
                err = T_pred_cmp[:, i] - T_meas_cmp[:, i]
                rmse = float(np.sqrt(np.mean(err ** 2)))
                bias = float(np.mean(err))
            else:
                rmse, bias = 0.0, 0.0

            ax.set_ylabel(f"{_NODE_LABELS[i]} [°C]")
            ax.legend(title=f"RMSE={rmse:.2f}°C  bias={bias:+.2f}°C",
                      loc="best", fontsize=7, title_fontsize=7)
            ax.grid(True, alpha=0.3)

        ax5 = axes[4]
        ax5.step(seg.index, Q_st_meas, where="post", color="tab:orange",
                 linewidth=1.2, label="Q_st meas")
        ax5.step(seg.index, Q_st_pred, where="post", color="tab:red",
                 linewidth=1.2, linestyle="--", label="Q_st pred")
        ax5.set_ylabel(f"Heat [kWh/{cfg.sampling_minutes}min]")
        ax5.legend(loc="upper right", fontsize=7)
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(bottom=0)

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
        axes[-1].set_xlabel("Time")
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()

        saved_path = None
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            tag = seg.index[0].strftime("%Y%m%d_%H%M")
            fname = output_dir / f"st_freeforward_{tag}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            saved_path = fname
            saved.append(fname)
            logger.info("Saved free-forward plot: %s", fname)
        plt.close(fig)

        # Terminal summary
        print(f"\nST Window {w.window_id}:")
        print(f"  Period      : {seg.index[0]} → {seg.index[-1]}  ({N} intervals)")
        print(f"  Q_st meas   : {float(Q_st_meas.sum()):.3f} kWh")
        print(f"  Q_st pred   : {float(Q_st_pred.sum()):.3f} kWh")
        if N > 1:
            print("  Free-forward RMSE (excluding reset):")
            for i, lbl in enumerate(_NODE_LABELS):
                err = T_pred_cmp[:, i] - T_meas_cmp[:, i]
                print(f"    {lbl:<8}: RMSE={float(np.sqrt(np.mean(err**2))):.3f}°C"
                      f"  bias={float(np.mean(err)):+.3f}°C")

    return saved
