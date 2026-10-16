"""Findhorn wind generation loader (FWP_Generation_2019.csv).

CSV schema:
    Date,Time stamp,FWP Generation (kWh)
    01/01/2019,00:00:00,78.03
    01/01/2019,01:00:00,36.67

Hourly kWh-per-hour (numerically equal to mean kW over the hour). We:
  1. Combine Date + Time stamp into a tz-naive Europe/London DatetimeIndex.
  2. Treat the value as average kW for that hour.
  3. Up-sample to 30-min by forward-fill (constant kW within the hour).
  4. Re-map onto the simulation index by (month, day, hour, minute) so a
     2024 run uses the matching 2019 day-of-year/hour pattern.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _load_raw_30min_kw(csv_path: Path) -> pd.Series:
    """Load the 2019 Findhorn Wind Park CSV and return a 30-min kW series."""
    raw = pd.read_csv(csv_path)
    expected = {"Date", "Time stamp", "FWP Generation (kWh)"}
    missing = expected - set(raw.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}; got {list(raw.columns)}")

    ts = pd.to_datetime(
        raw["Date"].astype(str) + " " + raw["Time stamp"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
    )
    hourly_kw = pd.Series(
        raw["FWP Generation (kWh)"].astype(float).values,
        index=ts,
        name="wind_kw",
    ).sort_index()

    # Hourly kWh equals mean kW over the hour. Up-sample to 30 min by ffill.
    end = hourly_kw.index.max() + pd.Timedelta(minutes=30)
    new_idx = pd.date_range(hourly_kw.index.min(), end, freq="30min", inclusive="left")
    return hourly_kw.reindex(new_idx, method="ffill")


def load_wind_mapped(
    csv_path: Path,
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    """Load 2019 wind generation and remap onto ``target_index`` by DOY+H+M.

    Returns a Series of kW aligned to ``target_index``. Leap-day Feb 29 in
    the target falls back to Feb 28 of the source year.
    """
    src = _load_raw_30min_kw(csv_path)

    src_key = list(zip(src.index.month, src.index.day, src.index.hour, src.index.minute))
    lookup = dict(zip(src_key, src.to_numpy()))

    out = np.full(len(target_index), np.nan, dtype=float)
    for i, ts in enumerate(target_index):
        key = (ts.month, ts.day, ts.hour, ts.minute)
        if key in lookup:
            out[i] = lookup[key]
        elif ts.month == 2 and ts.day == 29:
            out[i] = lookup.get((2, 28, ts.hour, ts.minute), np.nan)
    return pd.Series(out, index=target_index, name="wind_kw")
