"""
ASHP_fitting.evaluate – Diagnostic plotting and QC for ASHP fitting results.

For sufficiently long ASHP-only windows this module plots measured node
temperatures so the user can visually inspect thermal behaviour during
ASHP charging periods.  It also produces a QC summary CSV of back-calculated
Q and COP per interval for manual inspection.
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

from ASHP_fitting.config import ASHPFitConfig
from ASHP_fitting.detector import ASHPWindow

logger = logging.getLogger(__name__)

# Friendly node labels for plots
_NODE_LABELS = ["Bottom", "Mid", "Mid-Hi", "Top"]
_NODE_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


# ---------------------------------------------------------------------------
# Public: QC CSV (back-calculation details per interval)
# ---------------------------------------------------------------------------

def write_backcalc_csv(
    bc_df: pd.DataFrame,
    *,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Write the back-calculation results to a CSV for manual inspection.

    Parameters
    ----------
    bc_df : pd.DataFrame
        Output of ``fit_ashp_maps.back_calculate_q_ashp``.
    output_path : Path, optional

    Returns
    -------
    bc_df (unchanged), for chaining.
    """
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        bc_df.to_csv(output_path, index=False)
        logger.info("Back-calc CSV saved to %s", output_path)
    return bc_df


# ---------------------------------------------------------------------------
# Public: Diagnostic plots of ASHP-only windows
# ---------------------------------------------------------------------------

def evaluate_ashp_fit(
    df: pd.DataFrame,
    windows: List[ASHPWindow],
    ashp_fit: Dict,
    cfg: Optional[ASHPFitConfig] = None,
    *,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """Plot measured node temperatures during the longest ASHP-only windows.

    For each selected window, creates a 4-panel figure (one per node) with
    measured temperatures annotated with ASHP electricity and back-calc Q.

    Parameters
    ----------
    df : pd.DataFrame
        Full cleaned DataFrame.
    windows : list[ASHPWindow]
        Accepted ASHP-only windows from the detector.
    ashp_fit : dict
        Result from ``fit_ashp``.
    cfg : ASHPFitConfig
    output_dir : Path, optional

    Returns
    -------
    saved_paths : list[Path]
    """
    if cfg is None:
        cfg = ASHPFitConfig()

    n_train = int(len(df) * cfg.train_frac)
    df_train = df.iloc[:n_train]

    # Select longest windows for visual diagnostics
    sorted_windows = sorted(windows, key=lambda w: w.n_intervals, reverse=True)
    selected = sorted_windows[: cfg.n_plot_windows]

    if not selected:
        logger.warning("No ASHP-only windows available for plotting.")
        return []

    saved_paths: List[Path] = []

    for w in selected:
        seg = df_train.iloc[w.indices[0]: w.indices[-1] + 1]
        time_idx = seg.index

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(
            f"ASHP Window {w.window_id}:  {w.start:%Y-%m-%d %H:%M} → "
            f"{w.end:%Y-%m-%d %H:%M}  ({w.n_intervals} intervals)",
            fontsize=12,
        )

        for i, ax in enumerate(axes):
            node_col = cfg.node_cols[i]
            ax.plot(
                time_idx, seg[node_col].values,
                "o-", color=_NODE_COLOURS[i], markersize=3, linewidth=1.2,
                label=f"Measured {_NODE_LABELS[i]}",
            )
            ax.set_ylabel("Temperature [°C]")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time")
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        fig.autofmt_xdate()
        fig.tight_layout()

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            fname = output_dir / f"ashp_window_{w.window_id:03d}.png"
            fig.savefig(fname, dpi=100, bbox_inches="tight")
            saved_paths.append(fname)
            logger.info("Saved plot: %s", fname)

        plt.close(fig)

    return saved_paths
