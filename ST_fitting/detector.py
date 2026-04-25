"""
ST_fitting.detector – Detect ST-only charging windows using bottom-node
peak logic on 30-minute data.

Window detection rules:
  START:  bottom node rises > bottom_rise_start_c  AND
          ST power > st_power_start_kw  AND
          ASHP off  AND  immersion off.
  CONTINUE:  bottom node still rising (dT > 0).
  END:    bottom node stopped rising for ``bottom_flat_close``
          consecutive intervals.

The window is trimmed so that only intervals where the bottom node is
still rising are included (degradation tail excluded).  Back-calculated
energy is computed from all 4 node temperatures across the trimmed window.
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
    """A contiguous block of ST-only charging intervals.

    Attributes
    ----------
    window_id : int
        Sequential identifier.
    start : pd.Timestamp
        Timestamp of the first interval in the window.
    end : pd.Timestamp
        Timestamp of the last interval in the window (trimmed to peak).
    n_intervals : int
        Number of intervals in the window.
    indices : np.ndarray
        Integer positional indices into the training DataFrame.
    """
    window_id: int
    start: pd.Timestamp
    end: pd.Timestamp
    n_intervals: int
    indices: np.ndarray


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def detect_st_windows(
    df: pd.DataFrame,
    cfg: Optional[STFitConfig] = None,
) -> tuple[List[STWindow], pd.DataFrame]:
    """Detect ST-only charging windows in the training portion of *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame (output of ``src.data_loader.load_and_clean``).
        Must contain columns for the 4 node temperatures, ``st_power_kw``,
        ``ashp_inst_kwh``, and ``imm_tot_inst_kwh``.
    cfg : STFitConfig, optional
        Configuration object.  Defaults to ``STFitConfig()`` if not provided.

    Returns
    -------
    windows : list[STWindow]
        Validated ST-only windows trimmed to the bottom-node peak.
    diagnostics_df : pd.DataFrame
        One row per training interval with gate columns and window membership.
    """
    if cfg is None:
        cfg = STFitConfig()

    # -- 1. Training slice ---------------------------------------------------
    n_train = int(len(df) * cfg.train_frac)
    df_train = df.iloc[:n_train].copy()
    logger.info(
        "Training slice: %s → %s  (%d intervals, %d-min cadence)",
        df_train.index.min(), df_train.index.max(),
        len(df_train), cfg.sampling_minutes,
    )

    # -- 2. Bottom-node dT ---------------------------------------------------
    bottom = df_train[cfg.node_cols[0]].values
    dT_bottom = np.diff(bottom, prepend=np.nan)
    bottom_rising = dT_bottom > 0.0

    # -- 3. Gate signals -----------------------------------------------------
    g_power = df_train[cfg.st_power_col].fillna(0.0).values > cfg.st_power_start_kw
    g_rise = dT_bottom > cfg.bottom_rise_start_c
    g_ashp = df_train["ashp_inst_kwh"].fillna(0.0).values <= cfg.ashp_off_kwh
    g_imm = df_train["imm_tot_inst_kwh"].fillna(0.0).values <= cfg.imm_off_kwh

    can_open = g_power & g_rise & g_ashp & g_imm

    logger.info(
        "Gate pass rates — power: %d  rise>%.0f°C: %d  ashp_off: %d  "
        "imm_off: %d  can_open: %d  (of %d)",
        g_power.sum(), cfg.bottom_rise_start_c, g_rise.sum(),
        g_ashp.sum(), g_imm.sum(), can_open.sum(), len(df_train),
    )

    # -- 4. State machine: open / continue / close ---------------------------
    windows: List[STWindow] = []
    wid = 0
    in_window = False
    win_start = 0
    flat_count = 0  # consecutive non-rising bottom intervals

    for k in range(len(df_train)):
        if not in_window:
            if can_open[k]:
                in_window = True
                win_start = k
                flat_count = 0
        else:
            # Continuation: bottom still rising?
            if bottom_rising[k]:
                flat_count = 0
            else:
                flat_count += 1
                if flat_count >= cfg.bottom_flat_close:
                    # Close: trim to last rising interval
                    win_end = k - flat_count
                    _emit_window(windows, wid, win_start, win_end,
                                 df_train, cfg, bottom)
                    if windows and windows[-1].window_id == wid:
                        wid += 1
                    in_window = False
                    flat_count = 0

    # Handle window open at EOF
    if in_window:
        win_end = len(df_train) - 1 - flat_count
        _emit_window(windows, wid, win_start, win_end, df_train, cfg, bottom)

    total = sum(w.n_intervals for w in windows)
    logger.info(
        "Detected %d ST windows (%d intervals total)",
        len(windows), total,
    )

    # -- 5. Diagnostics DataFrame --------------------------------------------
    accepted = np.zeros(len(df_train), dtype=bool)
    for w in windows:
        accepted[w.indices] = True

    diag = pd.DataFrame({
        "time": df_train.index,
        "st_power_kw": df_train[cfg.st_power_col].values,
        "bottom_c": bottom,
        "dT_bottom": dT_bottom,
        "bottom_rising": bottom_rising,
        "g_power": g_power,
        "g_rise_start": g_rise,
        "g_ashp_off": g_ashp,
        "g_imm_off": g_imm,
        "in_window": accepted,
    })

    return windows, diag


def _emit_window(
    windows: List[STWindow],
    wid: int,
    win_start: int,
    win_end: int,
    df_train: pd.DataFrame,
    cfg: STFitConfig,
    bottom: np.ndarray,
) -> None:
    """Append a window if it meets the minimum-length criterion."""
    win_end = max(win_end, win_start)
    n_int = win_end - win_start + 1
    if n_int < cfg.min_st_intervals:
        return
    windows.append(STWindow(
        window_id=wid,
        start=df_train.index[win_start],
        end=df_train.index[win_end],
        n_intervals=n_int,
        indices=np.arange(win_start, win_end + 1),
    ))
