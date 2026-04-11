"""
ST_fitting.detector – Detect clean ST-only intervals in the dataset.

An interval is accepted when ALL four gates are true *and* the preceding
interval was also accepted (2-step persistence).  The gates are:

  G1  ST flow temp − tank bottom temp > st_flow_dt_min_c  [°C]
  G2  ST flow volume > st_flow_min_l                       [L/interval]
  G3  ST power > st_power_min_kw                           [kW]
  G4  tank bottom temp[k] > tank bottom temp[k−1]          (bottom rising)
  G5  ASHP energy <= ashp_off_kwh                          [kWh/interval]
  G6  immersion energy <= imm_off_kwh                      [kWh/interval]

Opening persistence rule (2 consecutive valid to open):
  valid[k]  = G1[k] & G2[k] & G3[k] & G4[k] & G5[k] & G6[k]
  valid[k]  = valid[k] & valid[k−1]

Closing persistence rule (2 consecutive invalid to close):
  A window that is open stays open through a single invalid interval;
  it closes only when two consecutive intervals both fail the gates.

This rejects curtailed ST episodes where the pump runs and power/flow
registers but no heat enters the store (e.g. buffer bypass, anti-steam
curtailment): those events pass G2 and G3 but fail G1 and/or G4 because
the temperature differential is absent.

NaN values in either the ST flow temp or tank bottom propagate as False
through G1, so curtailed intervals are excluded without any special NaN
branch.  An explicit NaN guard is still applied to the node and ambient
columns so that fitting residuals are never computed on incomplete data.

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
        outdoor temp, ``st_flow_temp_c``, ``st_flow_l``, ``st_power_kw``,
        ``ashp_inst_kwh``, and ``imm_tot_inst_kwh``.
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

    # -- Step 2: NaN guard for columns needed by the fitting model -----------
    required_cols = cfg.node_cols + [cfg.t_amb_col, cfg.t_out_col]
    has_nan = df_train[required_cols].isna().any(axis=1)

    # Also need finite temperatures at the *previous* row for dT calculation
    T = df_train[cfg.node_cols].values
    finite_now = np.all(np.isfinite(T), axis=1)
    finite_prev = np.roll(finite_now, 1)
    finite_prev[0] = False
    has_finite_pair = pd.Series(finite_now & finite_prev, index=df_train.index)

    # -- Step 3: Four detection gates ----------------------------------------
    # G1: ST flow temperature minus tank bottom > threshold
    #     NaN in either column → difference is NaN → gate is False (fillna)
    dT_flow = (
        df_train[cfg.st_flow_temp_col] - df_train[cfg.node_cols[0]]
    )
    g1 = (dT_flow > cfg.st_flow_dt_min_c).fillna(False)

    # G2: ST flow volume above minimum
    g2 = (df_train[cfg.st_flow_col].fillna(0.0) > cfg.st_flow_min_l)

    # G3: ST power above minimum
    g3 = (df_train[cfg.st_power_col].fillna(0.0) > cfg.st_power_min_kw)

    # G4: tank bottom temperature rising (first row is always False)
    g4 = (df_train[cfg.node_cols[0]].diff() > 0).fillna(False)

    # G5: ASHP off
    g5 = (df_train["ashp_inst_kwh"].fillna(0.0) <= cfg.ashp_off_kwh)

    # G6: immersion off
    g6 = (df_train["imm_tot_inst_kwh"].fillna(0.0) <= cfg.imm_off_kwh)

    logger.info(
        "Gate pass rates — G1: %d  G2: %d  G3: %d  G4: %d  G5: %d  G6: %d  (of %d intervals)",
        g1.sum(), g2.sum(), g3.sum(), g4.sum(), g5.sum(), g6.sum(), len(df_train),
    )

    # -- Step 4: Combine gates then apply 2-step opening persistence --------
    # All six gates must be true
    raw_valid = (g1.values & g2.values & g3.values & g4.values
                 & g5.values & g6.values)

    # Opening: also require the previous interval to have passed
    valid_prev = np.roll(raw_valid, 1)
    valid_prev[0] = False
    valid = raw_valid & valid_prev

    # -- Step 5: Apply NaN guard ---------------------------------------------
    accepted_arr = valid & (~has_nan).values & has_finite_pair.values

    # -- Step 6: Build per-interval diagnostics DataFrame --------------------
    diag = pd.DataFrame({
        "time": df_train.index,
        "g1_flow_dt": g1.values,
        "g2_flow_vol": g2.values,
        "g3_power": g3.values,
        "g4_bottom_rising": g4.values,
        "g5_ashp_off": g5.values,
        "g6_imm_off": g6.values,
        "no_nan": (~has_nan).values,
        "finite_pair": has_finite_pair.values,
        "accepted": accepted_arr,
    })

    # Assign rejection reason (first failing condition)
    reasons = []
    for _, row in diag.iterrows():
        if row["accepted"]:
            reasons.append("")
            continue
        if not row["g1_flow_dt"]:
            reasons.append("flow_dt_low")
        elif not row["g2_flow_vol"]:
            reasons.append("flow_vol_low")
        elif not row["g3_power"]:
            reasons.append("power_low")
        elif not row["g4_bottom_rising"]:
            reasons.append("bottom_not_rising")
        elif not row["g5_ashp_off"]:
            reasons.append("ashp_on")
        elif not row["g6_imm_off"]:
            reasons.append("imm_on")
        elif not row["no_nan"]:
            reasons.append("has_nan")
        elif not row["finite_pair"]:
            reasons.append("no_finite_pair")
        else:
            reasons.append("persistence")
    diag["reject_reason"] = reasons

    n_accepted = int(accepted_arr.sum())
    logger.info(
        "Accepted ST-only intervals: %d / %d (%.1f%%)",
        n_accepted, len(df_train), 100.0 * n_accepted / len(df_train),
    )

    # -- Step 7: Segment contiguous accepted intervals into windows ----------
    # Opening: first accepted interval after 2-step persistence (already baked
    #          into accepted_arr).
    # Closing: window closes only when TWO consecutive intervals are invalid
    #          (hysteresis — a single blip does not end the window).
    windows: List[STWindow] = []
    wid = 0

    in_window = False
    win_start = 0
    pending_close = False   # True after the first invalid interval inside a window

    for k in range(len(accepted_arr)):
        if accepted_arr[k]:
            if not in_window:
                win_start = k
                in_window = True
            pending_close = False   # reset: valid interval cancels a pending close
        else:
            if in_window:
                if not pending_close:
                    # First invalid — arm the close but keep the window open
                    pending_close = True
                else:
                    # Second consecutive invalid — close the window
                    # The window ends at k-2 (last accepted interval before the
                    # first invalid that armed the close)
                    win_end = k - 2
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
                    pending_close = False

    # Handle window still open at end of data
    if in_window:
        # If a close was pending the last valid interval was at len-2 (or len-1
        # if the very last sample was invalid but only one in a row)
        win_end = (len(accepted_arr) - 2) if pending_close else (len(accepted_arr) - 1)
        win_end = max(win_end, win_start)   # guard against degenerate case
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
    total_intervals_in_windows = sum(w.n_intervals for w in windows)
    logger.info(
        "ST-only windows: %d  (total intervals in windows: %d)",
        len(windows), total_intervals_in_windows,
    )

    return windows, diag
