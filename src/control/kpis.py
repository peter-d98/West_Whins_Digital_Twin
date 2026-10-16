"""KPI helpers for the predictive-control study.

Two tracks:

  1. **Measured-data KPIs** (``measured_kpis``): aggregate the raw plant
     CSV over a slice and split ASHP electricity into DHW vs SH using a
     heuristic. Used for sanity-checking the simulated baseline.

  2. **Simulator-output KPIs** (``simulated_kpis``): aggregate a
     ``SimResult`` from ``src.control.simulator``.

Common KPI columns (£ at Agile inc. VAT; kWh):
    ashp_kwh, imm_kwh, st_kwh, total_cost_gbp,
    wind_self_consumption_pct (or NaN if no wind data),
    n_ashp_fires, n_imm_fires, comfort_top_below_45_steps,
    comfort_mid_below_40_steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.control.simulator import SimResult


# ---------------------------------------------------------------------------
# SH-vs-DHW split heuristic for measured ASHP electricity
# ---------------------------------------------------------------------------
def split_ashp_dhw_vs_sh(
    df: pd.DataFrame,
    *,
    summer_assume_sh_zero: bool = True,
    mid_rise_threshold_c_per_step: float = 0.1,
    imm_off_threshold_kwh: float = 0.02,
    st_off_threshold_kw: float = 0.1,
) -> pd.Series:
    """Attribute ASHP electricity to DHW vs SH.

    Returns a Series ``ashp_dhw_kwh`` aligned to ``df`` (per-step kWh).

    Rule (winter): step contributes to DHW iff
        (mid_c rising > threshold) AND (imm ≈ off) AND (ST ≈ off).
    Summer (Jun-Aug): assume SH=0 → all ASHP elec is DHW.
    """
    if "ashp_inst_kwh" not in df.columns:
        raise ValueError("df missing ashp_inst_kwh")

    dhw = df["ashp_inst_kwh"].fillna(0.0).astype(float).copy()

    months = df.index.month
    summer_mask = np.asarray(months.isin([6, 7, 8]))

    if not summer_assume_sh_zero or (~summer_mask).any():
        mid = df["tank_mid_c"].astype(float)
        mid_rise = mid.diff().fillna(0.0)
        imm = df.get("imm_tot_inst_kwh", pd.Series(0.0, index=df.index)).fillna(0.0)
        st  = df.get("st_power_kw", pd.Series(0.0, index=df.index)).fillna(0.0)

        is_dhw_step = (
            (mid_rise > mid_rise_threshold_c_per_step)
            & (imm.abs() < imm_off_threshold_kwh)
            & (st.abs() < st_off_threshold_kw)
        ).to_numpy()
        # winter rows: keep ASHP only where is_dhw_step
        winter_mask = ~summer_mask
        mask = winter_mask & ~is_dhw_step
        dhw.iloc[mask] = 0.0

    return dhw.rename("ashp_dhw_kwh")


# ---------------------------------------------------------------------------
# Wind self-consumption metric
# ---------------------------------------------------------------------------
def wind_self_consumption_pct(
    e_ashp_kwh: pd.Series,
    wind_kw: Optional[pd.Series],
    *,
    dt_h: float = 0.5,
) -> float:
    """Fraction of ASHP electricity coincident with wind generation.

    Defined as Σ min(E_ashp_kwh_t, wind_kwh_t) / Σ E_ashp_kwh_t * 100.
    Returns NaN if wind data unavailable.
    """
    if wind_kw is None or e_ashp_kwh.sum() <= 0:
        return float("nan")
    wind_kwh = (wind_kw * dt_h).reindex(e_ashp_kwh.index).fillna(0.0)
    overlap = np.minimum(e_ashp_kwh.values, wind_kwh.values).sum()
    return float(100.0 * overlap / e_ashp_kwh.sum())


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------
def cost_gbp(
    e_kwh: pd.Series,
    price_p_per_kwh: pd.Series,
) -> float:
    """Σ E[t] · price[t] / 100  →  £."""
    aligned = price_p_per_kwh.reindex(e_kwh.index)
    if aligned.isna().any():
        # forward-fill any small gaps
        aligned = aligned.ffill().bfill()
    return float((e_kwh * aligned).sum() / 100.0)


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------
@dataclass
class KPIRow:
    label: str
    ashp_kwh: float
    imm_kwh: float
    st_kwh: float
    total_cost_gbp: float
    wind_self_consumption_pct: float
    n_ashp_fires: int
    n_imm_fires: int
    comfort_top_below_45_steps: int
    comfort_mid_below_40_steps: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def measured_kpis(
    df: pd.DataFrame,
    price_p_per_kwh: pd.Series,
    wind_kw: Optional[pd.Series],
    label: str,
    *,
    summer_assume_sh_zero: bool = True,
) -> KPIRow:
    """Compute KPIs from raw measured plant data over the slice."""
    ashp_dhw = split_ashp_dhw_vs_sh(df, summer_assume_sh_zero=summer_assume_sh_zero)
    imm = df.get("imm_tot_inst_kwh", pd.Series(0.0, index=df.index)).fillna(0.0)
    st_kw = df.get("st_power_kw", pd.Series(0.0, index=df.index)).fillna(0.0)
    st_kwh = (st_kw.clip(lower=0.0) * 0.5).sum()  # 30-min steps

    cost = cost_gbp(ashp_dhw, price_p_per_kwh) + cost_gbp(imm, price_p_per_kwh)
    wsc = wind_self_consumption_pct(ashp_dhw + imm, wind_kw)

    # "Fires" not directly observable in measured data — count contiguous
    # ASHP-DHW activity blocks as a proxy.
    fires_ashp = int(((ashp_dhw > 0.05) & (ashp_dhw.shift(1, fill_value=0.0) <= 0.05)).sum())
    fires_imm = int(((imm > 0.05) & (imm.shift(1, fill_value=0.0) <= 0.05)).sum())

    top = df.get("tank_top_c", pd.Series(np.nan, index=df.index))
    mid = df.get("tank_mid_c", pd.Series(np.nan, index=df.index))
    n_top_low = int((top < 45.0).sum())
    n_mid_low = int((mid < 40.0).sum())

    return KPIRow(
        label=label,
        ashp_kwh=float(ashp_dhw.sum()),
        imm_kwh=float(imm.sum()),
        st_kwh=float(st_kwh),
        total_cost_gbp=cost,
        wind_self_consumption_pct=wsc,
        n_ashp_fires=fires_ashp,
        n_imm_fires=fires_imm,
        comfort_top_below_45_steps=n_top_low,
        comfort_mid_below_40_steps=n_mid_low,
    )


def simulated_kpis(
    res: SimResult,
    price_p_per_kwh: pd.Series,
    wind_kw: Optional[pd.Series],
    label: str,
) -> KPIRow:
    """Compute KPIs from a simulator output."""
    e_ashp = pd.Series(res.E_ashp_kwh, index=res.times)
    e_imm = pd.Series(res.E_imm_kwh, index=res.times)
    cost = cost_gbp(e_ashp, price_p_per_kwh) + cost_gbp(e_imm, price_p_per_kwh)
    wsc = wind_self_consumption_pct(e_ashp + e_imm, wind_kw)

    top_seq = res.T_hist[1:, 3]
    mid_seq = res.T_hist[1:, 1]

    return KPIRow(
        label=label,
        ashp_kwh=float(e_ashp.sum()),
        imm_kwh=float(e_imm.sum()),
        st_kwh=float(res.Q_st_kwh.sum()),
        total_cost_gbp=cost,
        wind_self_consumption_pct=wsc,
        n_ashp_fires=int(res.ashp_flags.sum()),
        n_imm_fires=int(res.imm_flags.sum()),
        comfort_top_below_45_steps=int((top_seq < 45.0).sum()),
        comfort_mid_below_40_steps=int((mid_seq < 40.0).sum()),
    )


def kpis_to_frame(rows: list[KPIRow]) -> pd.DataFrame:
    return pd.DataFrame([r.to_dict() for r in rows])
