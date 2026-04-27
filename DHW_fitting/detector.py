"""
DHW_fitting.detector — Detect DHW draw events from drops in the tank
bottom-node temperature.

A draw event is a single interval where the bottom-node temperature
drops by more than ``draw_delta_c``.  At 30-minute resolution multi-
interval draws (e.g. a long shower) may merge into one event; the
volume back-calculation already accounts for the full ΔT recorded in
that interval.

The detection uses no ASHP/ST gating: a real draw can occur during
ASHP charging (charging would otherwise be raising T_bottom, so a
genuine drop overrides the charge effect and is still indicative of
demand).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import DhwFitConfig, V_BOTTOM_L

logger = logging.getLogger(__name__)


def detect_draw_events(df: pd.DataFrame, cfg: DhwFitConfig) -> pd.DataFrame:
    """Detect bottom-node draw events and back-calculate draw volumes.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned plant DataFrame with a DatetimeIndex and a
        ``tank_bottom_c`` column (canonical name from src.data_loader).
    cfg : DhwFitConfig
        Configuration with detection threshold and back-calc parameters.

    Returns
    -------
    pd.DataFrame
        One row per detected draw event with columns:

        - ``timestamp``      — DatetimeIndex value at the event interval
        - ``month``          — calendar month (1–12)
        - ``slot_of_day``    — half-hour slot index (0–47)
        - ``t_bottom_prev_c`` — bottom-node temperature at start of step [°C]
        - ``dT_bottom_c``    — measured drop (negative) [°C]
        - ``V_draw_l``       — back-calculated draw volume [L]
    """
    if "tank_bottom_c" not in df.columns:
        raise KeyError("DataFrame missing required column 'tank_bottom_c'")

    t_bot = df["tank_bottom_c"].to_numpy(dtype=float)
    t_prev = t_bot[:-1]
    t_curr = t_bot[1:]
    dT = t_curr - t_prev

    timestamps_all = df.index[1:]
    months_all = timestamps_all.month.to_numpy()

    # --- (B) Per-month baseline conduction-rise subtraction ----------------
    # Estimate the typical "quiet" dT_bottom in each month from rows where
    # the change is in a small band around zero (excludes draws AND charging
    # spikes).  Subtract this from dT before thresholding so that small
    # draws masked by conduction warming are still detectable.
    baseline_by_month = {m: 0.0 for m in range(1, 13)}
    if cfg.use_baseline_subtraction:
        lo, hi = cfg.baseline_quiet_band_c
        for m in range(1, 13):
            mask = (
                (months_all == m)
                & np.isfinite(dT)
                & (dT > lo)
                & (dT < hi)
            )
            if mask.sum() >= 20:
                baseline_by_month[m] = float(np.median(dT[mask]))
        logger.info(
            "Baseline conduction rise per month [°C/step]: %s",
            {m: round(v, 4) for m, v in baseline_by_month.items()},
        )

    baseline_arr = np.array([baseline_by_month[m] for m in months_all])
    dT_eff = dT - baseline_arr  # effective drop after removing conduction warming

    # Flag draw intervals: effective bottom-node drop exceeds the threshold.
    flagged = dT_eff < cfg.draw_delta_c

    # Numerical stability gate: need a meaningful gap above mains for
    # the volume back-calculation V = |dT_eff| * V_b / (T_prev - T_mains).
    gap = t_prev - cfg.t_mains_c
    stable = gap > cfg.min_temp_gap_c

    keep = flagged & stable & np.isfinite(dT_eff) & np.isfinite(t_prev)

    if not keep.any():
        logger.warning("No draw events detected (threshold %.2f °C).", cfg.draw_delta_c)
        return pd.DataFrame(
            columns=[
                "timestamp", "month", "slot_of_day",
                "t_bottom_prev_c", "dT_bottom_c", "dT_eff_c", "V_draw_l",
            ]
        )

    # Indices into df (events are recorded at the END of the falling step,
    # i.e. df.index[1:][keep] aligns with t_curr).
    event_idx = np.where(keep)[0] + 1
    timestamps = df.index[event_idx]

    t_prev_keep = t_prev[keep]
    dT_keep = dT[keep]
    dT_eff_keep = dT_eff[keep]
    gap_keep = gap[keep]

    # Back-calculate volume using bottom-node mass balance, using the
    # *effective* (conduction-corrected) drop:
    #   V_d = |dT_eff| * V_b / (T_prev - T_mains)
    V_draw = np.abs(dT_eff_keep) * V_BOTTOM_L / gap_keep
    V_draw_unclipped = V_draw.copy()
    V_draw = np.clip(V_draw, 0.0, cfg.v_draw_max_l)

    months = timestamps.month
    slots = timestamps.hour * 2 + timestamps.minute // 30

    out = pd.DataFrame({
        "timestamp": timestamps,
        "month": months.astype(int),
        "slot_of_day": slots.astype(int),
        "t_bottom_prev_c": t_prev_keep,
        "dT_bottom_c": dT_keep,
        "dT_eff_c": dT_eff_keep,
        "V_draw_l": V_draw,
    })

    n_events = len(out)
    n_clipped = int(np.sum(V_draw_unclipped > cfg.v_draw_max_l))
    logger.info(
        "Detected %d draw events (%d clipped at V_max=%.1f L).",
        n_events, n_clipped, cfg.v_draw_max_l,
    )
    return out
