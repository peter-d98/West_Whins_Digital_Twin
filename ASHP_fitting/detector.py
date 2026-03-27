"""
ASHP_fitting.detector – Detect clean ASHP-only intervals in the dataset.

An *ASHP-only interval* is one where:
  1. ASHP is ON            (ashp_inst_kwh > ashp_off_kwh)
  2. Solar-thermal is OFF  (st_kwh <= st_off_kwh)
  3. Immersion is OFF      (imm_tot_inst_kwh <= imm_off_kwh)
  4. No NaN values in node temperatures, ambient, or outdoor temp
  5. No draw event in THIS interval (bottom-node drop ≤ draw_delta_c)

Draw rejection is *interval-only*: a draw in one interval rejects that
interval but does NOT disqualify the entire surrounding window.  The
remaining contiguous accepted intervals are re-segmented into windows
after the draw intervals are removed.

Contiguous accepted intervals are grouped into windows.  Windows shorter
than ``min_ashp_intervals`` are rejected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from ASHP_fitting.config import ASHPFitConfig

logger = logging.getLogger(__name__)

# Column expected from solar_thermal.compute_st_energy (added externally)
_ST_COL = "st_kwh"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ASHPWindow:
    """A contiguous block of clean ASHP-only intervals.

    Attributes
    ----------
    window_id : int
        Sequential identifier.
    start : pd.Timestamp
        Timestamp of the first interval in the window.
    end : pd.Timestamp
        Timestamp of the last interval in the window.
    n_intervals : int
        Number of intervals in the window.
    indices : np.ndarray
        Integer positional indices into the training DataFrame for each
        interval in this window.
    """
    window_id: int
    start: pd.Timestamp
    end: pd.Timestamp
    n_intervals: int
    indices: np.ndarray   # positional iloc indices into df_train


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def detect_ashp_windows(
    df: pd.DataFrame,
    cfg: Optional[ASHPFitConfig] = None,
) -> tuple[List[ASHPWindow], pd.DataFrame]:
    """Detect ASHP-only windows in the training portion of *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame (output of ``src.data_loader.load_and_clean``).
        Must contain columns for the 4 node temperatures, ambient temp,
        outdoor temp, ``ashp_inst_kwh``, ``imm_tot_inst_kwh``, and ``st_kwh``.
    cfg : ASHPFitConfig, optional
        Configuration object.  Defaults to ``ASHPFitConfig()`` if not provided.

    Returns
    -------
    windows : list[ASHPWindow]
        Validated ASHP-only windows ready for back-calculation.
    diagnostics_df : pd.DataFrame
        One row per interval in the training set with boolean mask columns
        and a ``reject_reason`` column (empty string for accepted intervals).
    """
    if cfg is None:
        cfg = ASHPFitConfig()

    # -- Step 1: Extract training portion ------------------------------------
    n_train = int(len(df) * cfg.train_frac)
    df_train = df.iloc[:n_train].copy()
    logger.info(
        "Training slice: %s → %s  (%d intervals)",
        df_train.index.min(), df_train.index.max(), len(df_train),
    )

    # -- Step 2: Build boolean masks for each condition ----------------------
    ashp_on = df_train["ashp_inst_kwh"].fillna(0.0) > cfg.ashp_off_kwh
    st_off = df_train[_ST_COL].fillna(0.0) <= cfg.st_off_kwh
    imm_off = df_train["imm_tot_inst_kwh"].fillna(0.0) <= cfg.imm_off_kwh

    logger.info(
        "ASHP-on intervals: %d / %d (%.1f%%)",
        ashp_on.sum(), len(df_train), 100.0 * ashp_on.sum() / len(df_train),
    )

    # -- Step 3: NaN check ---------------------------------------------------
    required_cols = cfg.node_cols + [cfg.t_amb_col, cfg.t_out_col]
    has_nan = df_train[required_cols].isna().any(axis=1)

    # Also need finite temperatures at the *previous* row for dT calculation
    T = df_train[cfg.node_cols].values
    finite_now = np.all(np.isfinite(T), axis=1)
    finite_prev = np.roll(finite_now, 1)
    finite_prev[0] = False
    has_finite_pair = pd.Series(finite_now & finite_prev, index=df_train.index)

    # -- Step 4: Draw detection (interval-only rejection) --------------------
    bottom = df_train[cfg.node_cols[0]]
    bottom_diff = bottom.diff()
    is_draw = bottom_diff < cfg.draw_delta_c
    logger.info("Draw events in training data: %d", is_draw.sum())

    # -- Step 5: Combine into per-interval accepted mask ---------------------
    accepted = ashp_on & st_off & imm_off & ~has_nan & has_finite_pair & ~is_draw

    # -- Step 6: Build per-interval diagnostics DataFrame --------------------
    diag = pd.DataFrame({
        "time": df_train.index,
        "ashp_on": ashp_on.values,
        "st_off": st_off.values,
        "imm_off": imm_off.values,
        "no_nan": (~has_nan).values,
        "finite_pair": has_finite_pair.values,
        "no_draw": (~is_draw).values,
        "accepted": accepted.values,
    })

    # Assign rejection reason (first failing condition)
    reasons = []
    for _, row in diag.iterrows():
        if row["accepted"]:
            reasons.append("")
            continue
        if not row["ashp_on"]:
            reasons.append("ashp_off")
        elif not row["st_off"]:
            reasons.append("st_on")
        elif not row["imm_off"]:
            reasons.append("imm_on")
        elif not row["no_nan"]:
            reasons.append("has_nan")
        elif not row["finite_pair"]:
            reasons.append("no_finite_pair")
        elif not row["no_draw"]:
            reasons.append("draw_event")
        else:
            reasons.append("unknown")
    diag["reject_reason"] = reasons

    n_accepted = accepted.sum()
    logger.info(
        "Accepted ASHP-only intervals: %d / %d (%.1f%%)",
        n_accepted, len(df_train), 100.0 * n_accepted / len(df_train),
    )

    # -- Step 7: Segment contiguous accepted intervals into windows ----------
    accepted_arr = accepted.values
    windows: List[ASHPWindow] = []
    wid = 0

    in_window = False
    win_start = 0
    for k in range(len(accepted_arr)):
        if accepted_arr[k]:
            if not in_window:
                win_start = k
                in_window = True
        else:
            if in_window:
                win_end = k - 1
                n_int = win_end - win_start + 1
                if n_int >= cfg.min_ashp_intervals:
                    windows.append(ASHPWindow(
                        window_id=wid,
                        start=df_train.index[win_start],
                        end=df_train.index[win_end],
                        n_intervals=n_int,
                        indices=np.arange(win_start, win_end + 1),
                    ))
                    wid += 1
                in_window = False

    # Handle window that extends to end of data
    if in_window:
        win_end = len(accepted_arr) - 1
        n_int = win_end - win_start + 1
        if n_int >= cfg.min_ashp_intervals:
            windows.append(ASHPWindow(
                window_id=wid,
                start=df_train.index[win_start],
                end=df_train.index[win_end],
                n_intervals=n_int,
                indices=np.arange(win_start, win_end + 1),
            ))
            wid += 1

    # -- Step 8: Summary logging ---------------------------------------------
    n_draw_rejected = int(is_draw.sum())
    n_short = 0  # count windows rejected for being too short
    # Re-count short windows by examining all contiguous accepted+rejected blocks
    # (not needed here since we only create windows >= min_ashp_intervals)
    total_intervals_in_windows = sum(w.n_intervals for w in windows)
    logger.info(
        "ASHP-only windows: %d  (total intervals in windows: %d) | "
        "Draw-rejected intervals: %d",
        len(windows), total_intervals_in_windows, n_draw_rejected,
    )

    return windows, diag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_diagnostics() -> pd.DataFrame:
    """Return an empty diagnostics DataFrame with the expected columns."""
    return pd.DataFrame(columns=[
        "time", "ashp_on", "st_off", "imm_off", "no_nan",
        "finite_pair", "no_draw", "accepted", "reject_reason",
    ])
