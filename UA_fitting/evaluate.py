"""
UA_fitting.evaluate – Diagnostic plotting and QC CSV for UA fitting results.

For each plotted idle window this module:
  1. Takes the measured initial temperatures T[0] as the starting state.
  2. Simulates forward using ``tank_model.tank_step`` with Q_st=Q_ashp=Q_imm=0
     and only standing-loss dynamics (using the fitted UA_loss values, while
     keeping all other TankParams at defaults).
  3. Overlays measured vs predicted node temperatures.
  4. Reports RMSE per node.

This lets us visually and numerically verify that the fitted UA values
reproduce the observed cooling behaviour during idle periods.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI use
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from UA_fitting.config import UAConfig
from UA_fitting.detector import IdleWindow

# Import from the main codebase (read-only)
from src.tank_model import TankParams, tank_step

logger = logging.getLogger(__name__)

# Friendly node labels for plots
_NODE_LABELS = ["Bottom", "Mid", "Mid-Hi", "Top"]
_NODE_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


# ---------------------------------------------------------------------------
# Simulation helper
# ---------------------------------------------------------------------------

def _simulate_idle(
    window: IdleWindow,
    ua_loss: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    """Simulate standing-loss-only evolution for one idle window.

    Uses the fitted UA_loss values; all heat inputs set to zero, inter-node
    conduction and mixing use TankParams defaults (same baseline as fitting).

    Parameters
    ----------
    window : IdleWindow
    ua_loss : array of 4 UA_loss values [kW/K].
    dt_s : time-step in seconds.

    Returns
    -------
    T_pred : array (n_intervals, 4) — predicted temperatures at each step.
             Row 0 is the measured initial state (same as window.T_nodes[0]).
    """
    params = TankParams()
    params.UA_loss = np.array(ua_loss, dtype=float)
    # Zero out inter-node conduction (UA should account for these already)
    params.UA_adj = np.zeros(3)
    
    n = window.n_intervals
    T_pred = np.zeros((n, 4))
    T_pred[0] = window.T_nodes[0]  # initial condition from measurements

    for t in range(n - 1):
        T_pred[t + 1] = tank_step(
            T_pred[t],
            Q_st_kwh=0.0,
            Q_ashp_kwh=0.0,
            Q_imm_kwh=0.0,
            T_amb=window.T_amb[t],
            params=params,
            dt_s=dt_s,
        )

    return T_pred


def _compute_rmse(measured: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Per-node RMSE [°C] between measured and predicted, shape (4,)."""
    return np.sqrt(np.mean((measured - predicted) ** 2, axis=0))


# ---------------------------------------------------------------------------
# Public: QC CSV
# ---------------------------------------------------------------------------

def write_qc_csv(
    windows: List[IdleWindow],
    ua_loss: np.ndarray,
    cfg: Optional[UAConfig] = None,
    *,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Produce a QC summary CSV with per-window residual metrics.

    Parameters
    ----------
    windows : list[IdleWindow]
    ua_loss : array of 4 UA_loss values.
    cfg : UAConfig
    output_path : Path, optional — if given, CSV is written to disk.

    Returns
    -------
    qc_df : pd.DataFrame with columns:
        window_id, start, end, n_intervals, rmse_bottom, rmse_mid,
        rmse_mid_hi, rmse_top, mean_residual_bottom, ..., mean_residual_top.
    """
    if cfg is None:
        cfg = UAConfig()

    dt_s = cfg.sampling_minutes * 60.0
    rows = []

    for w in windows:
        T_pred = _simulate_idle(w, ua_loss, dt_s)
        rmse = _compute_rmse(w.T_nodes, T_pred)
        mean_resid = np.mean(w.T_nodes - T_pred, axis=0)

        rows.append({
            "window_id": w.window_id,
            "start": w.start,
            "end": w.end,
            "n_intervals": w.n_intervals,
            "rmse_bottom": round(float(rmse[0]), 4),
            "rmse_mid": round(float(rmse[1]), 4),
            "rmse_mid_hi": round(float(rmse[2]), 4),
            "rmse_top": round(float(rmse[3]), 4),
            "mean_resid_bottom": round(float(mean_resid[0]), 4),
            "mean_resid_mid": round(float(mean_resid[1]), 4),
            "mean_resid_mid_hi": round(float(mean_resid[2]), 4),
            "mean_resid_top": round(float(mean_resid[3]), 4),
        })

    qc_df = pd.DataFrame(rows)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        qc_df.to_csv(output_path, index=False)
        logger.info("QC CSV saved to %s", output_path)

    return qc_df


# ---------------------------------------------------------------------------
# Public: Diagnostic plots
# ---------------------------------------------------------------------------

def plot_qc(
    windows: List[IdleWindow],
    ua_fit: Dict,
    cfg: Optional[UAConfig] = None,
    *,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """Plot measured vs predicted temperatures for representative idle windows.

    Selects up to ``cfg.n_plot_windows`` windows that fall in summer months
    (configurable).  For each window, creates a 4-panel figure (one per node)
    showing measured and predicted temperatures with RMSE annotation.

    Parameters
    ----------
    windows : list[IdleWindow]
    ua_fit : dict with key ``"UA_loss"`` (array of 4 values).
    cfg : UAConfig
    output_dir : Path, optional — where to save PNG files.

    Returns
    -------
    saved_paths : list[Path] — paths to saved plot files.
    """
    if cfg is None:
        cfg = UAConfig()

    ua_loss = np.array(ua_fit["UA_loss"])
    dt_s = cfg.sampling_minutes * 60.0

    # -- Select summer windows -----------------------------------------------
    summer = [
        w for w in windows
        if w.start.month in cfg.summer_months
    ]

    if not summer:
        # Fall back to any available windows if no summer ones exist
        logger.warning(
            "No idle windows in summer months %s; falling back to all windows.",
            cfg.summer_months,
        )
        summer = list(windows)

    # Pick the longest N windows for best visual diagnostics
    summer.sort(key=lambda w: w.n_intervals, reverse=True)
    selected = summer[: cfg.n_plot_windows]

    if not selected:
        logger.warning("No windows available for plotting.")
        return []

    saved_paths: List[Path] = []

    for w in selected:
        T_pred = _simulate_idle(w, ua_loss, dt_s)
        rmse = _compute_rmse(w.T_nodes, T_pred)

        # Build a time axis for x-ticks
        time_idx = pd.date_range(
            start=w.start,
            periods=w.n_intervals,
            freq=f"{cfg.sampling_minutes}min",
        )

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(
            f"Idle Window {w.window_id}:  {w.start:%Y-%m-%d %H:%M} → "
            f"{w.end:%Y-%m-%d %H:%M}  ({w.n_intervals} intervals)",
            fontsize=12,
        )

        for i, ax in enumerate(axes):
            ax.plot(
                time_idx, w.T_nodes[:, i],
                "o-", color=_NODE_COLOURS[i], markersize=3, linewidth=1.2,
                label=f"Measured {_NODE_LABELS[i]}",
            )
            ax.plot(
                time_idx, T_pred[:, i],
                "x--", color=_NODE_COLOURS[i], markersize=4, linewidth=1.0,
                alpha=0.7,
                label=f"Predicted {_NODE_LABELS[i]}",
            )
            ax.set_ylabel("Temperature [°C]")
            ax.legend(loc="upper right", fontsize=8)
            ax.annotate(
                f"RMSE = {rmse[i]:.3f} °C",
                xy=(0.02, 0.05), xycoords="axes fraction",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.5),
            )
            ax.grid(True, alpha=0.3)

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        axes[-1].set_xlabel("Time")
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            fname = f"idle_window_{w.window_id}_{w.start:%Y%m%d_%H%M}.png"
            path = output_dir / fname
            fig.savefig(path, dpi=150)
            saved_paths.append(path)
            logger.info("Saved plot: %s", path)

        plt.close(fig)

    # Print summary RMSE across all plotted windows
    if selected:
        all_rmse = []
        for w in selected:
            T_p = _simulate_idle(w, ua_loss, dt_s)
            all_rmse.append(_compute_rmse(w.T_nodes, T_p))
        mean_rmse = np.mean(all_rmse, axis=0)
        logger.info(
            "Mean RMSE across %d plotted windows [°C]:  %s",
            len(selected),
            {_NODE_LABELS[i]: round(float(mean_rmse[i]), 4) for i in range(4)},
        )

    return saved_paths
