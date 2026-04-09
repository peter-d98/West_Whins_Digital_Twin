"""
ST_fitting.evaluate – Diagnostic plotting for ST fitting results.

Produces a free forward simulation plot for a selected ST-only window,
comparing measured vs predicted tank temperatures.  The simulation uses
measured ST energy as the heat input and sets ASHP / immersion to zero
(since the window is ST-only).

Also writes diagnostic CSVs for manual inspection.
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
from src.tank_model import TankParams, simulate

logger = logging.getLogger(__name__)

# Friendly node labels and colours (bottom→top)
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
    """Write the per-interval diagnostics to CSV.

    Parameters
    ----------
    diag_df : pd.DataFrame
        Output of ``detector.detect_st_windows`` (second return value).
    output_path : Path, optional

    Returns
    -------
    diag_df (unchanged), for chaining.
    """
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
    """Write the per-window summary to CSV.

    Parameters
    ----------
    window_df : pd.DataFrame
        Output of ``fit_st.fit_f_st`` (second return value).
    output_path : Path, optional

    Returns
    -------
    window_df (unchanged), for chaining.
    """
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        window_df.to_csv(output_path, index=False)
        logger.info("Window summary CSV saved to %s", output_path)
    return window_df


# ---------------------------------------------------------------------------
# Public: Free forward simulation plot for a selected ST window
# ---------------------------------------------------------------------------

def plot_st_window(
    df: pd.DataFrame,
    window: STWindow,
    tank_params: TankParams,
    cfg: Optional[STFitConfig] = None,
    *,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Free forward simulation over a single ST-only window.

    Resets to measured temperatures at window start, then simulates
    continuously using measured ST energy and the supplied tank parameters.
    ASHP and immersion are zero throughout (ST-only window).

    Parameters
    ----------
    df : pd.DataFrame
        Full cleaned DataFrame (must include ``st_kwh``).
    window : STWindow
        The selected window to plot.
    tank_params : TankParams
        Fitted tank parameters (from ``global_fit.json``).
    cfg : STFitConfig
    output_dir : Path, optional
        Directory for saved plot.

    Returns
    -------
    saved_path : Path or None
    """
    if cfg is None:
        cfg = STFitConfig()

    dt_s = cfg.sampling_minutes * 60.0
    dt_h = cfg.sampling_minutes / 60.0

    # -- Extract the window slice (including one row before for reset) -------
    n_train = int(len(df) * cfg.train_frac)
    df_train = df.iloc[:n_train]

    # Extend window by a small margin before/after for visual context.
    # The simulation starts at the first interval's prior row.
    first_k = window.indices[0]
    last_k = window.indices[-1]

    # Grab data from first_k to last_k (the window itself)
    seg = df_train.iloc[first_k: last_k + 1].copy()
    N = len(seg)

    if N < 2:
        logger.warning("Window %d has only %d rows; skipping plot.", window.window_id, N)
        return None

    node_cols = cfg.node_cols
    T_meas = seg[node_cols].values          # (N, 4)
    Q_st = seg["st_kwh"].fillna(0.0).values  # (N,)
    T_amb = seg[cfg.t_amb_col].fillna(seg[cfg.t_amb_col].median()).values

    # ST-only window: ASHP and immersion are zero
    Q_ashp = np.zeros(N)
    Q_imm = np.zeros(N)

    # Initial condition: measured state at [first_k - 1] (pre-window)
    if first_k > 0:
        T0 = df_train[node_cols].iloc[first_k - 1].values.astype(float)
    else:
        T0 = T_meas[0].copy()

    # -- Free forward simulation ---------------------------------------------
    T_hist = simulate(T0, Q_st, Q_ashp, Q_imm, T_amb, tank_params, dt_s=dt_s)
    # T_hist shape (N+1, 4); T_hist[0] = T0, T_hist[k] = state after k steps
    T_pred = T_hist[1:]    # (N, 4) — aligned with seg.index

    # RMSE against measured (excluding reset point = index 0 if T0 was prior row)
    T_meas_cmp = T_meas[1:]
    T_pred_cmp = T_pred[:N - 1]

    # -- Build figure --------------------------------------------------------
    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 2, 2, 1.2]})
    fig.suptitle(
        "Free forward simulation (ST charging window)\n"
        f"{seg.index[0]:%Y-%m-%d %H:%M} → {seg.index[-1]:%Y-%m-%d %H:%M}"
        f"  ({N} intervals)",
        fontsize=12,
    )

    # ST-on shading (all intervals in this window have ST on)
    dt_index = seg.index[1] - seg.index[0] if N > 1 else pd.Timedelta(minutes=cfg.sampling_minutes)

    # Temperature panels (indices 0–3)
    for i in range(4):
        ax = axes[i]
        ax.plot(seg.index, T_meas[:, i],
                "o-", color=_NODE_COLOURS[i], linewidth=1.4, markersize=3.5,
                label="Measured", zorder=3)
        ax.plot(seg.index, T_pred[:, i],
                "--", color=_NODE_COLOURS[i], linewidth=1.4, alpha=0.8,
                label="Predicted", zorder=3)
        # ST-on shading across entire window
        ax.axvspan(seg.index[0], seg.index[-1] + dt_index,
                   alpha=0.10, color="tab:orange", zorder=1)

        # RMSE and bias (skip first point which is the reset)
        if N > 1:
            err = T_pred_cmp[:, i] - T_meas_cmp[:, i]
            rmse = float(np.sqrt(np.mean(err ** 2)))
            bias = float(np.mean(err))
        else:
            rmse, bias = 0.0, 0.0

        ax.set_ylabel(f"{_NODE_LABELS[i]} [°C]")
        ax.legend(
            title=f"RMSE={rmse:.2f}°C  bias={bias:+.2f}°C",
            loc="best", fontsize=7, title_fontsize=7,
        )
        ax.grid(True, alpha=0.3)

    # Heat input panel (index 4)
    ax5 = axes[4]
    ax5.step(seg.index, Q_st, where="post", color="tab:orange",
             linewidth=1.2, label="Q_st")
    ax5.set_ylabel(f"Heat input\n[kWh / {cfg.sampling_minutes} min]")
    ax5.legend(loc="upper right", fontsize=7)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(bottom=0)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
    axes[-1].set_xlabel("Time")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    # -- Save ----------------------------------------------------------------
    saved_path = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tag = seg.index[0].strftime("%Y%m%d_%H%M")
        fname = output_dir / f"st_window_{tag}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        saved_path = fname
        logger.info("Saved ST window plot: %s", fname)

    plt.close(fig)

    # -- Print terminal summary ----------------------------------------------
    print(f"\nST Window {window.window_id}:")
    print(f"  Period     : {seg.index[0]} → {seg.index[-1]}  ({N} intervals)")
    print(f"  Q_st total : {float(Q_st.sum()):.3f} kWh")
    if N > 1:
        print("  Free-simulation RMSE (excluding reset point):")
        for i, lbl in enumerate(_NODE_LABELS):
            err = T_pred_cmp[:, i] - T_meas_cmp[:, i]
            print(f"    {lbl:<8}: RMSE={float(np.sqrt(np.mean(err**2))):.3f}°C"
                  f"  bias={float(np.mean(err)):+.3f}°C")

    return saved_path
