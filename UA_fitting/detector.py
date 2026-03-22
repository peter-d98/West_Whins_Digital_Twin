"""
UA_fitting.detector – Detect idle tank periods in the cleaned dataset.

An *idle window* is a contiguous stretch of intervals where:
  1. ASHP is off           (ashp_inst_kwh <= threshold)
  2. Solar-thermal is off  (st_kwh <= threshold)
  3. Immersion is off      (imm_tot_inst_kwh <= threshold)
  4. No draw events        (bottom-node drop ≤ draw_delta_c)
  5. No NaN values in node temperatures or ambient
  6. Window length >= min_idle_intervals

When all heat sources are off and no draw occurs, the tank temperature
evolution is governed solely by standing losses to the surrounding
plant-room air, which is exactly the regime we want for UA fitting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from UA_fitting.config import UAConfig

logger = logging.getLogger(__name__)

# Column expected from solar_thermal.compute_st_energy (added externally)
_ST_COL = "st_kwh"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class IdleWindow:
    """A single validated idle window ready for UA fitting.

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
    T_nodes : np.ndarray
        Node temperatures, shape (n_intervals, 4) [°C].  Bottom→top.
    T_amb : np.ndarray
        Ambient (plant-room) temperatures, shape (n_intervals,) [°C].
    mean_dT : float
        Mean (T_node_avg - T_amb) over the window [°C].
    mean_amb : float
        Mean T_amb over the window [°C].
    """
    window_id: int
    start: pd.Timestamp
    end: pd.Timestamp
    n_intervals: int
    T_nodes: np.ndarray    # shape (n_intervals, 4)
    T_amb: np.ndarray      # shape (n_intervals,)
    mean_dT: float
    mean_amb: float


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def detect_idle_windows(
    df: pd.DataFrame,
    cfg: Optional[UAConfig] = None,
) -> tuple[List[IdleWindow], pd.DataFrame]:
    """Detect idle windows in the training portion of *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame (output of ``src.data_loader.load_and_clean``).
        Must contain columns for the 4 node temperatures, ambient temp,
        ``ashp_inst_kwh``, ``imm_tot_inst_kwh``, and ``st_kwh``.
    cfg : UAConfig, optional
        Configuration object.  Defaults to ``UAConfig()`` if not provided.

    Returns
    -------
    windows : list[IdleWindow]
        Validated idle windows ready for fitting.
    diagnostics_df : pd.DataFrame
        One row per *candidate* window (before rejection) with summary
        statistics and rejection reason, suitable for saving as CSV.
    """
    if cfg is None:
        cfg = UAConfig()

    # -- Step 1: Extract training portion (first train_frac of data) ---------
    n_train = int(len(df) * cfg.train_frac)
    df_train = df.iloc[:n_train].copy()
    logger.info(
        "Training slice: %s → %s  (%d intervals)",
        df_train.index.min(), df_train.index.max(), len(df_train),
    )

    # -- Step 2: Build boolean masks for each "off" condition ----------------
    ashp_off = df_train["ashp_inst_kwh"].fillna(0.0) <= cfg.ashp_off_kwh
    st_off = df_train[_ST_COL].fillna(0.0) <= cfg.st_off_kwh
    imm_off = df_train["imm_tot_inst_kwh"].fillna(0.0) <= cfg.imm_off_kwh

    # Combined: all heat sources off
    all_off = ashp_off & st_off & imm_off
    logger.info(
        "Intervals with all heat sources off: %d / %d (%.1f%%)",
        all_off.sum(), len(df_train), 100.0 * all_off.sum() / len(df_train),
    )

    # -- Step 3: Detect draw events (sharp bottom-node drop) -----------------
    bottom = df_train[cfg.node_cols[0]]
    bottom_diff = bottom.diff()  # negative = temperature drop
    is_draw = bottom_diff < cfg.draw_delta_c  # True where a draw occurred
    logger.info("Draw events detected: %d", is_draw.sum())

    # -- Step 4: Check for NaN in required columns ---------------------------
    required_cols = cfg.node_cols + [cfg.t_amb_col]
    has_nan = df_train[required_cols].isna().any(axis=1)


    # -- Step 4b: Exclude intervals with sudden node temperature jumps -------
    # A jump is defined as any node increasing by more than 1.5°C in a single timestep
    node_diffs = df_train[cfg.node_cols].diff()  # shape (n, 4)
    jump_mask = (node_diffs > cfg.jump_thrsh).any(axis=1)  # True where any node jumps up >1.5°C

    # -- Step 5: Mark intervals that are fully "idle and clean" ---------------
    idle_mask = all_off & ~is_draw & ~has_nan & ~jump_mask

    # -- Step 6: Segment consecutive idle intervals into candidate windows ----
    #   We label contiguous True blocks by computing the cumulative sum of
    #   transitions from False→True, giving each block a unique group ID.
    block_id = (~idle_mask).cumsum()
    block_id[~idle_mask] = -1  # mark non-idle intervals

    candidate_groups = (
        block_id[block_id >= 0]
        .reset_index()
        .rename(columns={block_id.name or 0: "block"})
    )
    if candidate_groups.empty:
        logger.warning("No idle intervals found at all.")
        return [], _empty_diagnostics()

    # Group by contiguous block
    grouped = candidate_groups.groupby("block")

    # -- Step 7: Validate each candidate window and build output -------------
    windows: List[IdleWindow] = []
    diag_rows = []
    wid = 0

    for _, grp in grouped:
        start_ts = grp["time"].iloc[0]
        end_ts = grp["time"].iloc[-1]
        n_int = len(grp)

        # Extract temperature arrays for this window
        mask = (df_train.index >= start_ts) & (df_train.index <= end_ts)
        seg = df_train.loc[mask]

        # Count NaNs in required columns (should be 0 given prior filtering)
        n_nans = int(seg[required_cols].isna().sum().sum())

        # Check for remaining draw events inside the window
        seg_bottom_diff = seg[cfg.node_cols[0]].diff()
        contains_draw = bool((seg_bottom_diff < cfg.draw_delta_c).any())

        # Determine rejection reason
        reject_reason = ""
        if n_int < cfg.min_idle_intervals:
            reject_reason = f"too_short ({n_int} < {cfg.min_idle_intervals})"
        elif n_nans > 0:
            reject_reason = f"contains_nan ({n_nans})"
        elif contains_draw:
            reject_reason = "contains_draw"

        T_nodes = seg[cfg.node_cols].values   # (n_int, 4)
        T_amb = seg[cfg.t_amb_col].values     # (n_int,)
        node_mean = T_nodes.mean(axis=1)      # average across 4 nodes
        mean_dT = float(np.mean(node_mean - T_amb))
        mean_amb = float(np.mean(T_amb))

        diag_rows.append({
            "window_id": wid,
            "start": start_ts,
            "end": end_ts,
            "n_intervals": n_int,
            "mean_dT_c": round(mean_dT, 2),
            "mean_amb_c": round(mean_amb, 2),
            "n_nans": n_nans,
            "reject_reason": reject_reason,
        })

        if reject_reason == "":
            windows.append(IdleWindow(
                window_id=wid,
                start=start_ts,
                end=end_ts,
                n_intervals=n_int,
                T_nodes=T_nodes,
                T_amb=T_amb,
                mean_dT=mean_dT,
                mean_amb=mean_amb,
            ))

        wid += 1

    # -- Step 8: Summary logging ---------------------------------------------
    n_rejected_short = sum(1 for r in diag_rows if "too_short" in r["reject_reason"])
    n_rejected_nan = sum(1 for r in diag_rows if "contains_nan" in r["reject_reason"])
    n_rejected_draw = sum(1 for r in diag_rows if "contains_draw" in r["reject_reason"])
    logger.info(
        "Candidate windows: %d | Accepted: %d | "
        "Rejected (short: %d, NaN: %d, draw: %d)",
        len(diag_rows), len(windows),
        n_rejected_short, n_rejected_nan, n_rejected_draw,
    )

    diagnostics_df = pd.DataFrame(diag_rows)
    return windows, diagnostics_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_diagnostics() -> pd.DataFrame:
    """Return an empty diagnostics DataFrame with the expected columns."""
    return pd.DataFrame(columns=[
        "window_id", "start", "end", "n_intervals",
        "mean_dT_c", "mean_amb_c", "n_nans", "reject_reason",
    ])
