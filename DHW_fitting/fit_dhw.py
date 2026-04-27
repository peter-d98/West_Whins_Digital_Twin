"""
DHW_fitting.fit_dhw — Build a monthly DHW demand profile from detected
bottom-node draw events.

Profile design
--------------
For each (month, half-hour slot) pair the profile stores ``mean_V_l``,
the *expected value* of the draw volume — i.e. the mean over **all**
days in the learning set, including days with no draw at that slot
(zeros).  This gives the correct cumulative demand when the profile is
applied deterministically every simulated day.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DhwFitConfig
from .detector import detect_draw_events

logger = logging.getLogger(__name__)

N_SLOTS_PER_DAY = 48  # 30-minute cadence
N_MONTHS = 12


def _learning_slice(df: pd.DataFrame, train_frac: float) -> pd.DataFrame:
    """Return the first `train_frac` of df chronologically."""
    n = len(df)
    n_train = int(round(n * train_frac))
    if n_train < 2:
        raise ValueError(f"Learning slice too small: n_train={n_train}")
    return df.iloc[:n_train]


def build_profile(
    events: pd.DataFrame,
    df_train: pd.DataFrame,
    min_event_frequency: float = 0.0,
) -> pd.DataFrame:
    """Aggregate detected events into a (month, slot) demand profile.

    Parameters
    ----------
    events : pd.DataFrame
        Output of :func:`detect_draw_events` restricted to the learning slice.
    df_train : pd.DataFrame
        The learning-slice DataFrame (used to count the number of days
        per month present in the data, the denominator for the mean).
    min_event_frequency : float
        Suggestion (A): zero out (month, slot) entries where draws occur
        on fewer than this fraction of days (n_events / n_days).  Default 0
        (no filtering); 0.05 is a reasonable noise floor.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns
        ``[month, slot, time, mean_V_l, n_events, n_days, frequency]``
        covering every (month, slot) pair, even those with zero events.
    """
    # Total V_draw per (month, slot) — sum across all days.
    if len(events) > 0:
        sum_V = events.groupby(["month", "slot_of_day"])["V_draw_l"].sum()
        cnt_V = events.groupby(["month", "slot_of_day"]).size()
    else:
        sum_V = pd.Series(dtype=float)
        cnt_V = pd.Series(dtype=int)

    # Number of *valid* days per (month, slot): a day-slot is valid if
    # the bottom-node temperature change can be evaluated (both this row
    # and the previous row are finite).  This is the correct denominator
    # for the mean-including-zeros, because purely missing days otherwise
    # bias the mean downward.
    if "tank_bottom_c" not in df_train.columns:
        raise KeyError("df_train missing required column 'tank_bottom_c'")
    t_bot = df_train["tank_bottom_c"]
    valid_step = t_bot.notna() & t_bot.shift(1).notna()
    valid_idx = df_train.index[valid_step]
    valid_df = pd.DataFrame({
        "month": valid_idx.month.astype(int),
        "slot": (valid_idx.hour * 2 + valid_idx.minute // 30).astype(int),
        "date": valid_idx.normalize(),
    })
    # Unique (month, slot, date) combinations: count days that have a
    # valid observation at that slot.
    n_days_by_slot = (
        valid_df.drop_duplicates(["month", "slot", "date"])
                .groupby(["month", "slot"])
                .size()
    )

    rows = []
    n_floored = 0
    for month in range(1, N_MONTHS + 1):
        for slot in range(N_SLOTS_PER_DAY):
            key = (month, slot)
            total_V = float(sum_V.get(key, 0.0))
            n_events = int(cnt_V.get(key, 0))
            n_days = int(n_days_by_slot.get(key, 0))
            mean_V = total_V / n_days if n_days > 0 else 0.0
            frequency = (n_events / n_days) if n_days > 0 else 0.0
            # Suggestion (A): zero out slots where draws occur too rarely
            # to be statistically meaningful — likely noise rather than real demand.
            if min_event_frequency > 0.0 and 0.0 < frequency < min_event_frequency:
                if mean_V > 0.0:
                    n_floored += 1
                mean_V = 0.0
            time_str = f"{slot // 2:02d}:{(slot % 2) * 30:02d}"
            rows.append({
                "month": month,
                "slot": slot,
                "time": time_str,
                "mean_V_l": mean_V,
                "n_events": n_events,
                "n_days": n_days,
                "frequency": frequency,
            })
    if min_event_frequency > 0.0:
        logger.info(
            "Frequency floor %.2f: zeroed %d (month, slot) entries.",
            min_event_frequency, n_floored,
        )
    return pd.DataFrame(rows)


def main(cfg: DhwFitConfig, df: pd.DataFrame) -> dict:
    """Run the DHW profile pipeline and write outputs.

    Parameters
    ----------
    cfg : DhwFitConfig
    df : pd.DataFrame
        Full cleaned dataset; the learning slice is derived internally.

    Returns
    -------
    dict
        Summary statistics: total events, annual-equivalent volume/energy,
        learning-slice date range.
    """
    out_dir = Path(cfg.output_dir)
    diag_dir = Path(cfg.diagnostics_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    if cfg.use_all_data:
        df_train = df
        logger.info(
            "Using ENTIRE dataset for profile learning: %s → %s (%d rows)",
            df_train.index[0], df_train.index[-1], len(df_train),
        )
    else:
        df_train = _learning_slice(df, cfg.train_frac)
        logger.info(
            "Learning slice: %s → %s (%d rows, %.1f%% of total)",
            df_train.index[0], df_train.index[-1], len(df_train),
            100.0 * len(df_train) / len(df),
        )

    events = detect_draw_events(df_train, cfg)
    events_path = diag_dir / "draw_events.csv"
    events.to_csv(events_path, index=False)
    logger.info("Wrote raw draw events → %s", events_path)

    profile = build_profile(
        events, df_train, min_event_frequency=cfg.min_event_frequency,
    )
    profile_path = out_dir / "dhw_profile.csv"
    profile.to_csv(profile_path, index=False)
    logger.info("Wrote demand profile → %s", profile_path)

    # Annual equivalent: for each month sum mean_V_l across the 48 slots
    # to get the average daily demand for that month, then multiply by the
    # nominal calendar days in the month.
    DAYS_IN_MONTH = {
        1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
        7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31,
    }
    annual_V_l = 0.0
    per_month_daily_l = {}
    for month in range(1, N_MONTHS + 1):
        daily_mean = profile.loc[profile["month"] == month, "mean_V_l"].sum()
        per_month_daily_l[month] = float(daily_mean)
        annual_V_l += daily_mean * DAYS_IN_MONTH[month]

    # Approximate annual energy assuming DHW supply at ~55 °C from upper
    # node, mains at cfg.t_mains_c → ΔT = 55 - T_mains.  ρCp = 4.186 kJ/(L·K).
    delta_T_use = 55.0 - cfg.t_mains_c
    annual_kwh = annual_V_l * 4.186 * delta_T_use / 3600.0

    # --- (C) Summer-evening diagnostic --------------------------------------
    # Investigate whether evening (18:00–23:30, slots 36–47) zeros are due
    # to data quality, ST-recharge masking, or genuine low demand.
    evening_slots = list(range(36, 48))
    evening_diag_rows = []
    for month in range(1, N_MONTHS + 1):
        ev = events[(events["month"] == month)
                    & (events["slot_of_day"].isin(evening_slots))]
        sub_prof = profile[(profile["month"] == month)
                           & (profile["slot"].isin(evening_slots))]
        # Median dT_bottom over ALL evening rows that month (event or not)
        ev_mask = (
            (df_train.index.month == month)
            & df_train.index.hour.isin(range(18, 24))
        )
        dT_evening = df_train.loc[ev_mask, "tank_bottom_c"].diff().dropna()
        evening_diag_rows.append({
            "month": month,
            "n_evening_events": int(len(ev)),
            "n_evening_zero_slots": int((sub_prof["mean_V_l"] == 0).sum()),
            "median_dT_evening_c": float(dT_evening.median()) if len(dT_evening) else float("nan"),
            "mean_dT_evening_c": float(dT_evening.mean()) if len(dT_evening) else float("nan"),
        })
    evening_diag = pd.DataFrame(evening_diag_rows)
    evening_diag_path = diag_dir / "summer_evening_diag.csv"
    evening_diag.to_csv(evening_diag_path, index=False)
    logger.info("Wrote summer-evening diagnostic → %s", evening_diag_path)

    # --- Daily-volume stochastic stats per month ----------------------------
    # Total V_draw per (month, date), then mean/std across days within month
    daily_stats_rows = []
    if len(events) > 0:
        events_with_date = events.assign(date=events["timestamp"].dt.normalize())
        daily_totals = events_with_date.groupby(["month", "date"])["V_draw_l"].sum()
        for month in range(1, N_MONTHS + 1):
            if month in daily_totals.index.get_level_values(0):
                vals = daily_totals.loc[month]
                daily_stats_rows.append({
                    "month": month,
                    "n_days_with_draws": int(len(vals)),
                    "mean_daily_l": float(vals.mean()),
                    "std_daily_l": float(vals.std()),
                    "cv": float(vals.std() / vals.mean()) if vals.mean() > 0 else float("nan"),
                    "p05_daily_l": float(vals.quantile(0.05)),
                    "p95_daily_l": float(vals.quantile(0.95)),
                })
    daily_stats = pd.DataFrame(daily_stats_rows)
    daily_stats_path = out_dir / "dhw_daily_stats.csv"
    daily_stats.to_csv(daily_stats_path, index=False)
    logger.info("Wrote per-month daily-volume stats → %s", daily_stats_path)

    summary = {
        "n_events": int(len(events)),
        "learning_start": str(df_train.index[0]),
        "learning_end": str(df_train.index[-1]),
        "annual_volume_l": float(annual_V_l),
        "annual_energy_kwh": float(annual_kwh),
        "delta_T_use_c": float(delta_T_use),
        "per_month_daily_l": per_month_daily_l,
    }
    logger.info(
        "Total events: %d | Annual ≈ %.0f L (~%.0f kWh @ ΔT=%.0f °C)",
        summary["n_events"], summary["annual_volume_l"],
        summary["annual_energy_kwh"], delta_T_use,
    )
    return summary
