"""
Data loader and cleaner for the West Whins DHW digital twin.

Reads the plant CSV, applies column mapping from YAML, handles
cumulative-meter differencing, rollover repair, #N/A tokens,
negative PV clipping, and 30-min grid alignment.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def load_column_mapping(yaml_path: str | Path) -> dict:
    """Return the full mapping dictionary from *column_mapping.yaml*."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_and_clean(
    csv_path: str | Path,
    yaml_path: str | Path,
    *,
    sampling_minutes: int = 30,
) -> pd.DataFrame:
    """Load the plant CSV, clean, and return a tidy DataFrame.

    Parameters
    ----------
    csv_path : path to the raw CSV file.
    yaml_path : path to column_mapping.yaml.
    sampling_minutes : expected cadence (default 30).

    Returns
    -------
    pd.DataFrame with a ``DatetimeIndex`` named ``time`` and standardised
    column names (see ``_CANONICAL`` below).
    """
    cfg = load_column_mapping(yaml_path)

    # ---- 1. Read CSV, treating #N/A as NaN --------------------------------
    df = pd.read_csv(
        csv_path,
        na_values=["#N/A", "#N/A!", "#REF!", "N/A", ""],
        encoding="utf-8-sig",          # handles BOM
    )
    logger.info("Raw CSV shape: %s", df.shape)

    # ---- 2. Parse time and set as index -----------------------------------
    time_col = cfg["time"]["name"]
    time_fmt = cfg["time"]["format"]
    df[time_col] = pd.to_datetime(df[time_col], format=time_fmt, dayfirst=True)
    df = df.set_index(time_col).sort_index()
    df.index.name = "time"

    # Align to expected grid
    freq = f"{sampling_minutes}min"
    df = df.asfreq(freq)

    # ---- 3. Build canonical rename map ------------------------------------
    rename_map = _build_rename_map(cfg)
    # Keep only columns present in the data
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # ---- 4. Convert to numeric (coerce leftovers) -------------------------
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- 5. Cumulative → interval differencing ----------------------------
    df = _diff_cumulative(df, "ashp_elec_cum_kwh", "ashp_inst_kwh")
    df = _diff_cumulative(df, "imm_tot_cum_kwh", "imm_tot_inst_kwh")

    # ---- 6. Clip small negative PV at night to zero ----------------------
    if "pv_inst_kw" in df.columns:
        neg_pv = df["pv_inst_kw"] < 0
        if neg_pv.any():
            logger.info("Clipping %d negative PV values to 0.", neg_pv.sum())
            df.loc[neg_pv, "pv_inst_kw"] = 0.0

    # ---- 7. Temperature sanity clipping -----------------------------------
    temp_cols = ["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c", "t_amb_c"]
    for tc in temp_cols:
        if tc in df.columns:
            bad = (df[tc] < -10) | (df[tc] > 99)
            if bad.any():
                logger.info("Clipping %d implausible values in %s.", bad.sum(), tc)
                df.loc[bad, tc] = np.nan

    # ---- 8. Clip obviously wrong interval energies ------------------------
    for ec in ["ashp_inst_kwh", "imm_tot_inst_kwh", "st_power_kw"]:
        if ec in df.columns:
            neg = df[ec] < 0
            if neg.any():
                logger.info("Clipping %d negative values in %s.", neg.sum(), ec)
                df.loc[neg, ec] = 0.0

    # ---- 9. Forward-fill tiny gaps (≤2 steps) then leave NaN --------------
    df = df.ffill(limit=2)

    logger.info("Cleaned DataFrame shape: %s, date range %s → %s",
                df.shape, df.index.min(), df.index.max())
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Canonical name → YAML path tuples
_YAML_PATHS: list[tuple[str, list[str]]] = [
    ("t_amb_c",           ["ambient", "ambient_c"]),
    ("tank_bottom_c",     ["tank", "bottom_c"]),
    ("tank_mid_c",        ["tank", "mid_c"]),
    ("tank_mid_hi_c",     ["tank", "mid_hi_c"]),
    ("tank_top_c",        ["tank", "top_c"]),
    ("st_power_kw",       ["solar_thermal", "power_kw"]),
    ("st_flow_l",         ["solar_thermal", "flow_l"]),
    ("st_flow_temp_c",    ["solar_thermal", "flow_temp_c"]),
    ("st_return_temp_c",  ["solar_thermal", "return_temp_c"]),
    ("st_tot_energy_mwh", ["solar_thermal", "tot_energy_mwh"]),
    ("st_tot_vol_l",      ["solar_thermal", "tot_vol_l"]),
    ("ashp_elec_cum_kwh", ["ashp", "elec_cum_kwh"]),
    ("ashp_inst_kwh",     ["ashp", "elec_int_kwh"]),
    ("imm_tot_cum_kwh",   ["immersion", "total_cum_kwh"]),
    ("imm_tot_inst_kwh",  ["immersion", "total_int_kwh"]),
    ("backup_imm_kwh",    ["immersion", "backup_int_kwh"]),
    ("pv_inst_kw",        ["pv_proxy", "inst_kw"]),
]


def _build_rename_map(cfg: dict) -> dict[str, str]:
    """Return ``{csv_header: canonical_name}`` from the YAML mapping."""
    rename: dict[str, str] = {}
    for canon, keys in _YAML_PATHS:
        node = cfg
        try:
            for k in keys:
                node = node[k]
        except (KeyError, TypeError):
            continue
        if node is not None:
            rename[node] = canon
    return rename


def _diff_cumulative(
    df: pd.DataFrame,
    cum_col: str,
    int_col: str,
) -> pd.DataFrame:
    """Difference a cumulative meter column into interval kWh.

    If the interval column already exists **and** is non-trivial (>50 %
    non-zero) we keep it.  Otherwise we derive it from the cumulative
    column with rollover repair.
    """
    if cum_col not in df.columns:
        return df

    has_interval = (
        int_col in df.columns
        and df[int_col].notna().sum() > 0
        and (df[int_col].fillna(0) != 0).mean() > 0.05
    )

    if has_interval:
        # Still derive a backup for QC
        derived = df[cum_col].diff()
        # Rollover repair: negative diffs replaced with NaN
        derived = derived.where(derived >= 0, np.nan)
        df[f"{int_col}_derived"] = derived
    else:
        derived = df[cum_col].diff()
        derived = derived.where(derived >= 0, np.nan)
        df[int_col] = derived

    return df


def node_ordering_check(df: pd.DataFrame) -> pd.Series:
    """Return a boolean Series: True where T_top >= T_mh >= T_mid >= T_bot."""
    cols = ["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]
    if not all(c in df.columns for c in cols):
        return pd.Series(True, index=df.index)
    ordered = (
        (df["tank_top_c"] >= df["tank_mid_hi_c"] - 0.5)
        & (df["tank_mid_hi_c"] >= df["tank_mid_c"] - 0.5)
        & (df["tank_mid_c"] >= df["tank_bottom_c"] - 0.5)
    )
    return ordered
