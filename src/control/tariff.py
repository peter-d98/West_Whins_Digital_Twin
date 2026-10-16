"""Octopus Agile historical-tariff fetcher and loader.

Provides 30-min unit-rate prices (p/kWh, inc. VAT) on a UTC index, then
re-indexed to Europe/London local time matching the rest of the pipeline.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# Region E = North Scotland (Findhorn).
DEFAULT_PRODUCT = "AGILE-FLEX-22-11-25"
DEFAULT_TARIFF = "E-1R-AGILE-FLEX-22-11-25-E"
_API = "https://api.octopus.energy/v1/products/{product}/electricity-tariffs/{tariff}/standard-unit-rates/"


def _fetch_page(url: str, retries: int = 3, backoff: float = 1.5) -> dict:
    last_err: Exception | None = None
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as exc:  # network blip
            last_err = exc
            time.sleep(backoff ** (k + 1))
    raise RuntimeError(f"fetch failed: {url}") from last_err


def fetch_agile(
    period_from: pd.Timestamp,
    period_to: pd.Timestamp,
    product: str = DEFAULT_PRODUCT,
    tariff: str = DEFAULT_TARIFF,
) -> pd.DataFrame:
    """Fetch Agile half-hourly unit rates for [period_from, period_to) UTC.

    Returns a DataFrame indexed by UTC timestamp (start of 30-min slot)
    with a single column ``price_p_per_kwh`` (inc. VAT).
    """
    if period_from.tzinfo is None:
        period_from = period_from.tz_localize("UTC")
    if period_to.tzinfo is None:
        period_to = period_to.tz_localize("UTC")

    base = _API.format(product=product, tariff=tariff)
    url = (
        f"{base}?period_from={period_from.strftime('%Y-%m-%dT%H:%MZ')}"
        f"&period_to={period_to.strftime('%Y-%m-%dT%H:%MZ')}"
        f"&page_size=1500"
    )

    rows: list[dict] = []
    while url:
        page = _fetch_page(url)
        rows.extend(page["results"])
        url = page.get("next")

    if not rows:
        raise RuntimeError(f"no rates returned for {period_from}..{period_to}")

    df = pd.DataFrame(rows)
    df["valid_from"] = pd.to_datetime(df["valid_from"], utc=True)
    df = (
        df.rename(columns={"value_inc_vat": "price_p_per_kwh"})
        .set_index("valid_from")
        .sort_index()[["price_p_per_kwh"]]
    )
    return df


def fetch_and_save(
    out_path: Path,
    windows: list[tuple[str, str]],
    product: str = DEFAULT_PRODUCT,
    tariff: str = DEFAULT_TARIFF,
) -> pd.DataFrame:
    """Fetch one or more UTC windows and concatenate into a single CSV."""
    parts = []
    for start, end in windows:
        parts.append(
            fetch_agile(pd.Timestamp(start), pd.Timestamp(end), product, tariff)
        )
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    return df


def load_tariff(
    csv_path: Path,
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    """Load Agile prices and reindex onto a (tz-naive Europe/London) target.

    The CSV is in UTC; target_index is the simulation index, assumed to
    represent Europe/London local time (tz-naive, matching FullDS_Findhorn_*).
    Returns a Series of p/kWh aligned to target_index.
    """
    df = pd.read_csv(csv_path, parse_dates=["valid_from"], index_col="valid_from")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # Convert UTC rates to local time, then drop tz to match the sim index.
    local = df.tz_convert("Europe/London")
    local.index = local.index.tz_localize(None)

    # Forward-fill (Agile is on 30-min boundaries already; reindex handles align).
    return local["price_p_per_kwh"].reindex(target_index, method="nearest", tolerance=pd.Timedelta("30min"))


# ---------------------------------------------------------------------------
# Static day/night tariff (current West Whins operator pricing)
# ---------------------------------------------------------------------------
def static_day_night_tariff(
    target_index: pd.DatetimeIndex,
    *,
    day_p_per_kwh: float = 36.07,
    night_p_per_kwh: float = 31.71,
    day_start_hour: int = 7,
    day_end_hour: int = 12,
) -> pd.Series:
    """Return a static day/night tariff aligned to ``target_index``.

    Default values reflect the current operator tariff:
      * 07:00–12:00 → 36.07 p/kWh (day)
      * all other hours → 31.71 p/kWh (night)

    The boundary is half-open: a 30-min slot whose start-hour is in
    ``[day_start_hour, day_end_hour)`` pays the day rate; everything
    else pays the night rate.
    """
    hours = target_index.hour
    is_day = (hours >= day_start_hour) & (hours < day_end_hour)
    rates = np.where(is_day, day_p_per_kwh, night_p_per_kwh)
    return pd.Series(rates, index=target_index, name="price_p_per_kwh")


# ---------------------------------------------------------------------------
# Findhorn site-specific flex tariff (one full year, hourly, £/kWh)
# ---------------------------------------------------------------------------
def load_flex_tariff(
    csv_path: Path,
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    """Load the Findhorn flex-tariff CSV and map it onto ``target_index``.

    The CSV provides 8760 hourly rows (one calendar year, 2019) of
    £/kWh values under the column
    ``"NFD Import Tariff to End Users  (£/kWh)"``. We map by
    (month, day, hour) so the same intra-year shape can be applied to
    any simulation year (Feb 29 falls back to Feb 28).

    Returns a Series of **p/kWh** aligned to ``target_index``.
    """
    df = pd.read_csv(csv_path)
    price_col = [c for c in df.columns if "Tariff" in c][0]
    ts = pd.to_datetime(
        df["Date"] + " " + df["Time stamp"], dayfirst=True
    )
    df = df.assign(
        month=ts.dt.month, day=ts.dt.day, hour=ts.dt.hour,
        price=df[price_col].astype(float) * 100.0,  # £/kWh -> p/kWh
    )
    lut = df.set_index(["month", "day", "hour"])["price"]

    out = np.empty(len(target_index), dtype=float)
    for i, ts in enumerate(target_index):
        key = (int(ts.month), int(ts.day), int(ts.hour))
        if key not in lut.index:  # Feb 29 -> Feb 28 fallback
            key = (key[0], 28, key[2])
        out[i] = float(lut.loc[key])
    return pd.Series(out, index=target_index, name="price_p_per_kwh")
