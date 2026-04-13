"""
ST_fitting.evaluate – Diagnostic plotting and CSV output for ST fitting.

Provides:
  - write_interval_csv / write_window_csv : save detector diagnostics.
  - plot_regression_diagnostics : GTI vs Q_meas_kwh and T_flow scatter + fit.
  - plot_freeforward_windows : free-forward simulation over selected windows
    using the fitted Q_sol and T_flow maps with dynamic node allocation.
"""

from __future__ import annotations

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
from src.tank_model import NODE_CAP

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
    """Write the per-interval diagnostics to CSV."""
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
    """Write the per-window summary to CSV."""
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
    """Two-panel scatter diagnostics for Q_sol and T_flow regressions.

    Panel 1: Q_pred vs Q_meas with y=x reference line.
    Panel 2: GTI vs T_flow residuals (measured − predicted).

    Points are coloured by window_id.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Filter to rows with valid GTI
    plot_df = reg_df.dropna(subset=["GTI"]).copy()
    if plot_df.empty:
        logger.warning("No data with GTI for regression plot.")
        plt.close(fig)
        return None

    unique_wids = sorted(plot_df["window_id"].unique())
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(unique_wids), 1)))
    wid_to_colour = {wid: cmap[i % len(cmap)] for i, wid in enumerate(unique_wids)}
    colours = plot_df["window_id"].map(wid_to_colour).values

    # --- Panel 1: Q_sol prediction quality ---------------------------------
    q = result["Q_sol"]
    q_pred = plot_df["q_pred_kwh"].values
    q_meas = plot_df["q_meas_kwh"].values
    ax1.scatter(q_pred, q_meas, c=colours, s=14, alpha=0.7, edgecolors="none")
    q_max = float(np.nanmax(np.concatenate([q_pred, q_meas]))) if len(plot_df) else 1.0
    q_line = np.linspace(0, q_max * 1.05, 200)
    ax1.plot(q_line, q_line, "k--", linewidth=1.2)
    ax1.set_xlabel("Q_pred [kWh/interval]")
    ax1.set_ylabel("Q_meas [kWh/interval]")
    ax1.set_title(
        f"Q_sol multivariate fit (R² = {q['r2']:.3f}, n = {q['n_points']})  "
        f"q0={q['q0_kwh']:.4f}, q1={q['q1_kwh_per_wm2']:.6f}, "
        f"q2={q.get('q2_kwh_per_c', 0.0):.6f}, q3={q.get('q3_kwh_per_c', 0.0):.6f}",
        fontsize=9,
    )
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: T_flow residuals ------------------------------------------
    tf = result["T_flow"]
    t_pred = plot_df["t_flow_pred_c"]
    t_meas = plot_df["st_flow_temp_c"]
    valid_t = t_pred.notna() & t_meas.notna()
    resid = t_meas[valid_t] - t_pred[valid_t]
    ax2.scatter(plot_df.loc[valid_t, "GTI"], resid, c=np.array(colours)[valid_t.values], s=14, alpha=0.7, edgecolors="none")
    ax2.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("GTI [W/m²]")
    ax2.set_ylabel("T_flow residual (meas − pred) [°C]")
    ax2.set_title(
        f"T_flow = {tf['b0_c']:.2f} + {tf['b1_per_c']:.3f}·T_bot + {tf['b2_c_per_wm2']:.5f}·GTI  "
        f"(R² = {tf['r2']:.3f}, n = {tf['n_points']})",
        fontsize=9,
    )
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    saved_path = None
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        saved_path = output_path
        logger.info("Saved regression diagnostics plot: %s", saved_path)
    plt.close(fig)
    return saved_path


# ---------------------------------------------------------------------------
# Public: Free-forward validation plot per ST window
# ---------------------------------------------------------------------------

def plot_freeforward_windows(
    df: pd.DataFrame,
    windows: List[STWindow],
    result: Dict,
    cfg: Optional[STFitConfig] = None,
    *,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """Free-forward simulation over selected ST-only windows.

    For each window the simulation:
      1. Resets to measured temperatures at the pre-window row.
      2. Predicts Q_sol from the fitted energy map using measured GTI
         (loaded from ``reg_df`` via the ``result`` coefficients).
      3. Computes T_flow from the bilinear map.
      4. Allocates heat to nodes via dynamic weights.
      5. Advances the tank state (losses + conduction, ASHP/imm = 0).

    Because these are *identified* ST windows, Q_sol is always treated as
    active during the window (no GTI >= 180 / T_bottom < 55 gate).  This
    lets us validate the energy and allocation maps directly.
    """
    if cfg is None:
        cfg = STFitConfig()

    dt_s = cfg.sampling_minutes * 60.0
    dt_h = cfg.sampling_minutes / 60.0
    saved: List[Path] = []

    q = result["Q_sol"]
    tf = result["T_flow"]

    # We need GTI inside the windows.  Load the GTI CSV from the standard path.
    # Since this is only for plotting, we import lazily.
    from ST_fitting.fit_st import _load_gti
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    gti_csv = _REPO_ROOT / "data" / "hist_GTI_5min.csv"
    if gti_csv.exists():
        gti_series = _load_gti(gti_csv)
    else:
        logger.warning("GTI CSV not found; free-forward plots skipped.")
        return saved

    n_train = int(len(df) * cfg.train_frac)
    df_train = df.iloc[:n_train].copy()
    df_train["GTI"] = gti_series.reindex(df_train.index).fillna(0.0)

    # Load UA params from global_fit.json for losses/conduction
    import json
    global_json = _REPO_ROOT / "Global_fitting" / "output" / "global_fit.json"
    if global_json.exists():
        with open(global_json, "r", encoding="utf-8") as f:
            gj = json.load(f)
        UA_loss = np.array(gj["UA_loss"], dtype=float)
        UA_adj = np.array(gj["UA_adj"], dtype=float)
    else:
        logger.warning("global_fit.json not found; using default UA values for plots.")
        UA_loss = np.array([0.003, 0.002, 0.002, 0.003])
        UA_adj = np.array([0.05, 0.05, 0.05])

    node_cols = cfg.node_cols

    for w in windows:
        first_k = w.indices[0]
        last_k = w.indices[-1]
        seg = df_train.iloc[first_k: last_k + 1].copy()
        N = len(seg)
        if N < 2:
            continue

        T_meas = seg[node_cols].values          # (N, 4)
        T_amb = seg[cfg.t_amb_col].fillna(seg[cfg.t_amb_col].median()).values
        T_out = seg[cfg.t_out_col].fillna(seg[cfg.t_out_col].median()).values
        GTI_arr = seg["GTI"].values

        # Initial condition
        if first_k > 0:
            T0 = df_train[node_cols].iloc[first_k - 1].values.astype(float)
        else:
            T0 = T_meas[0].copy()

        # --- Free-forward simulation (ST always active during window) -------
        T_hist = np.zeros((N + 1, 4))
        T_hist[0] = T0

        for k in range(N):
            T_cur = T_hist[k]
            gti_k = float(GTI_arr[k])

            # Predict Q_sol and T_flow (always active — this is a known window)
            Q_sol_k = predict_q_sol_kwh(
                gti_k,
                q["q0_kwh"],
                q["q1_kwh_per_wm2"],
                t_bottom_c=float(T_cur[0]),
                t_out_c=float(T_out[k]),
                q2_kwh_per_c=float(q.get("q2_kwh_per_c", 0.0)),
                q3_kwh_per_c=float(q.get("q3_kwh_per_c", 0.0)),
            )
            t_flow_k = tf["b0_c"] + tf["b1_per_c"] * T_cur[0] + tf["b2_c_per_wm2"] * gti_k
            f_st_k = st_node_weights_from_tflow(T_cur, t_flow_k)

            Q_st_kj = Q_sol_k * 3600.0
            T_new = T_cur.copy()
            for i in range(4):
                dQ = f_st_k[i] * Q_st_kj
                loss = UA_loss[i] * (T_cur[i] - T_amb[k]) * dt_s
                cond = 0.0
                if i > 0:
                    cond += UA_adj[i - 1] * (T_cur[i - 1] - T_cur[i]) * dt_s
                if i < 3:
                    cond += UA_adj[i] * (T_cur[i + 1] - T_cur[i]) * dt_s
                dT = (dQ - loss + cond) / NODE_CAP[i]
                T_new[i] = T_cur[i] + dT

            T_new = np.clip(T_new, 5.0, 95.0)
            T_hist[k + 1] = T_new

        T_pred = T_hist[1:]  # (N, 4)

        # --- Build figure ---------------------------------------------------
        fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 2, 2, 2, 1.2]})
        fig.suptitle(
            "Free forward simulation (dynamic ST allocation)\n"
            f"{seg.index[0]:%Y-%m-%d %H:%M} → {seg.index[-1]:%Y-%m-%d %H:%M}"
            f"  ({N} intervals)",
            fontsize=12,
        )

        dt_index = seg.index[1] - seg.index[0] if N > 1 else pd.Timedelta(minutes=cfg.sampling_minutes)

        for i in range(4):
            ax = axes[i]
            ax.plot(seg.index, T_meas[:, i], "o-", color=_NODE_COLOURS[i],
                    linewidth=1.4, markersize=3.5, label="Measured", zorder=3)
            ax.plot(seg.index, T_pred[:, i], "--", color=_NODE_COLOURS[i],
                    linewidth=1.4, alpha=0.8, label="Predicted", zorder=3)
            ax.axvspan(seg.index[0], seg.index[-1] + dt_index,
                       alpha=0.10, color="tab:orange", zorder=1)

            if N > 1:
                err = T_pred[1:, i] - T_meas[1:, i]
                rmse = float(np.sqrt(np.mean(err ** 2)))
                bias = float(np.mean(err))
            else:
                rmse, bias = 0.0, 0.0

            ax.set_ylabel(f"{_NODE_LABELS[i]} [°C]")
            ax.legend(title=f"RMSE={rmse:.2f}°C  bias={bias:+.2f}°C",
                      loc="best", fontsize=7, title_fontsize=7)
            ax.grid(True, alpha=0.3)

        # Heat input panel
        ax5 = axes[4]
        Q_pred_kwh = np.array([
            predict_q_sol_kwh(
                float(GTI_arr[k]),
                q["q0_kwh"],
                q["q1_kwh_per_wm2"],
                t_bottom_c=float(T_meas[k, 0]),
                t_out_c=float(T_out[k]),
                q2_kwh_per_c=float(q.get("q2_kwh_per_c", 0.0)),
                q3_kwh_per_c=float(q.get("q3_kwh_per_c", 0.0)),
            )
            for k in range(N)
        ])
        Q_meas_kwh = seg[cfg.st_power_col].fillna(0.0).values * dt_h
        ax5.step(seg.index, Q_meas_kwh, where="post", color="tab:orange",
                 linewidth=1.2, label="Q_meas")
        ax5.step(seg.index, Q_pred_kwh, where="post", color="tab:blue",
                 linewidth=1.2, linestyle="--", label="Q_pred")
        ax5.set_ylabel(f"ST energy\n[kWh / {cfg.sampling_minutes} min]")
        ax5.legend(loc="upper right", fontsize=7)
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(bottom=0)

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
        axes[-1].set_xlabel("Time")
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            tag = seg.index[0].strftime("%Y%m%d_%H%M")
            fname = output_dir / f"st_freeforward_{tag}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            saved.append(fname)
            logger.info("Saved free-forward plot: %s", fname)

        plt.close(fig)

        # Summary
        print(f"\nST Window {w.window_id}:")
        print(f"  Period     : {seg.index[0]} → {seg.index[-1]}  ({N} intervals)")
        print(f"  Q_pred tot : {float(Q_pred_kwh.sum()):.3f} kWh")
        print(f"  Q_meas tot : {float(Q_meas_kwh.sum()):.3f} kWh")
        if N > 1:
            print("  Free-forward RMSE:")
            for i, lbl in enumerate(_NODE_LABELS):
                err = T_pred[1:, i] - T_meas[1:, i]
                print(f"    {lbl:<8}: RMSE={float(np.sqrt(np.mean(err**2))):.3f}°C"
                      f"  bias={float(np.mean(err)):+.3f}°C")

    return saved
