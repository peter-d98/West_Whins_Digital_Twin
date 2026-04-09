"""
ST_fitting.detector – Detect clean ST-only intervals in the dataset.

An *ST-only interval* is one where:
  1. Solar-thermal is ON   (st_kwh > st_on_kwh)
  2. ASHP is OFF           (ashp_inst_kwh <= ashp_off_kwh)
  3. Immersion is OFF      (imm_tot_inst_kwh <= imm_off_kwh)
  4. No NaN values in node temperatures, ambient, or outdoor temp
  5. No draw event in THIS interval (bottom-node drop ≤ draw_delta_c)

Draw rejection is *interval-only*: a draw in one interval rejects only that
interval.  The remaining contiguous accepted intervals are re-segmented
into windows after rejection.

Unlike the ASHP detector, no Schmitt trigger or mid-rising gate is needed.
The ST power measurement itself is the primary on-signal: when the ST
circulation pump runs and measurable heat is delivered, the interval qualifies.

Contiguous accepted intervals are grouped into windows.  Windows shorter
than ``min_st_intervals`` are rejected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from ST_fitting.config import STFitConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class STWindow:
    """A contiguous block of clean ST-only intervals.

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

def detect_st_windows(
    df: pd.DataFrame,
    cfg: Optional[STFitConfig] = None,
) -> tuple[List[STWindow], pd.DataFrame]:
    """Detect ST-only windows in the training portion of *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame (output of ``src.data_loader.load_and_clean``).
        Must contain columns for the 4 node temperatures, ambient temp,
        outdoor temp, ``ashp_inst_kwh``, ``imm_tot_inst_kwh``, and ``st_kwh``.
    cfg : STFitConfig, optional
        Configuration object.  Defaults to ``STFitConfig()`` if not provided.

    Returns
    -------
    windows : list[STWindow]
        Validated ST-only windows ready for f_st calculation.
    diagnostics_df : pd.DataFrame
        One row per interval in the training set with boolean mask columns
        and a ``reject_reason`` column (empty string for accepted intervals).
    """
    if cfg is None:
        cfg = STFitConfig()

    # -- Step 1: Extract training portion ------------------------------------
    n_train = int(len(df) * cfg.train_frac)
    df_train = df.iloc[:n_train].copy()
    logger.info(
        "Training slice: %s → %s  (%d intervals)",
        df_train.index.min(), df_train.index.max(), len(df_train),
    )

    # -- Step 2: Build per-interval condition masks --------------------------
    st_on = df_train["st_kwh"].fillna(0.0) > cfg.st_on_kwh
    ashp_off = df_train["ashp_inst_kwh"].fillna(0.0) <= cfg.ashp_off_kwh
    imm_off = df_train["imm_tot_inst_kwh"].fillna(0.0) <= cfg.imm_off_kwh

    logger.info(
        "ST-on intervals: %d / %d (%.1f%%)",
        st_on.sum(), len(df_train), 100.0 * st_on.sum() / len(df_train),
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

    # -- Step 5: Combine into accepted mask ----------------------------------
    accepted_arr = (
        st_on.values
        & ashp_off.values
        & imm_off.values
        & (~has_nan).values
        & has_finite_pair.values
        & (~is_draw).values
    )

    # -- Step 6: Build per-interval diagnostics DataFrame --------------------
    diag = pd.DataFrame({
        "time": df_train.index,
        "st_on": st_on.values,
        "ashp_off": ashp_off.values,
        "imm_off": imm_off.values,
        "no_nan": (~has_nan).values,
        "finite_pair": has_finite_pair.values,
        "no_draw": (~is_draw).values,
        "accepted": accepted_arr,
    })

    # Assign rejection reason (first failing condition)
    reasons = []
    for _, row in diag.iterrows():
        if row["accepted"]:
            reasons.append("")
            continue
        if not row["st_on"]:
            reasons.append("st_off")
        elif not row["ashp_off"]:
            reasons.append("ashp_on")
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

    n_accepted = int(accepted_arr.sum())
    logger.info(
        "Accepted ST-only intervals: %d / %d (%.1f%%)",
        n_accepted, len(df_train), 100.0 * n_accepted / len(df_train),
    )

    # -- Step 7: Segment contiguous accepted intervals into windows ----------
    windows: List[STWindow] = []
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
                if n_int >= cfg.min_st_intervals:
                    windows.append(STWindow(
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
        if n_int >= cfg.min_st_intervals:
            windows.append(STWindow(
                window_id=wid,
                start=df_train.index[win_start],
                end=df_train.index[win_end],
                n_intervals=n_int,
                indices=np.arange(win_start, win_end + 1),
            ))
            wid += 1

    # -- Step 8: Summary logging ---------------------------------------------
    n_draw_rejected = int(is_draw.sum())
    total_intervals_in_windows = sum(w.n_intervals for w in windows)
    logger.info(
        "ST-only windows: %d  (total intervals in windows: %d) | "
        "Draw-rejected intervals: %d",
        len(windows), total_intervals_in_windows, n_draw_rejected,
    )

    return windows, diag
