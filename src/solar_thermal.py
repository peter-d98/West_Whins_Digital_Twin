"""
Solar-thermal as-measured module.

Stage-1 uses measured ST power or derives it from flow and delta-T.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Physical constants
RHO = 1000.0        # kg/m³  (water density)
CP  = 4.186          # kJ/(kg·K)


def compute_st_energy(
    df: pd.DataFrame,
    *,
    dt_minutes: float = 30.0,
    hx_effectiveness: float = 1.0,
) -> pd.Series:
    """Return interval ST heat delivered to the DHW tank [kWh].

    Strategy
    --------
    1. If ``st_power_kw`` column is available and mostly populated, integrate
       it over the interval:  Q = P [kW] × (dt/60) [h].
    2. Otherwise derive from flow and temperatures:
       Q̇ = ρ · cp · V̇ · (T_flow − T_ret) · η_HX   →  kW.
       Then Q [kWh] = Q̇ × (dt/60).
    """
    dt_h = dt_minutes / 60.0

    has_power = (
        "st_power_kw" in df.columns
        and df["st_power_kw"].notna().mean() > 0.5
    )

    if has_power:
        q_kwh = df["st_power_kw"].fillna(0.0).clip(lower=0) * dt_h
        logger.info("ST energy from st_power_kw column.")
    else:
        q_kwh = _derive_from_flow(df, dt_h, hx_effectiveness)
        logger.info("ST energy derived from flow × ΔT.")

    q_kwh.name = "st_kwh"
    return q_kwh


def _derive_from_flow(
    df: pd.DataFrame,
    dt_h: float,
    hx_eff: float,
) -> pd.Series:
    """Derive ST kWh from flow rate and temperatures."""
    flow_l = df.get("st_flow_l", pd.Series(0.0, index=df.index)).fillna(0.0)
    t_flow = df.get("st_flow_temp_c", pd.Series(0.0, index=df.index)).fillna(0.0)
    t_ret  = df.get("st_return_temp_c", pd.Series(0.0, index=df.index)).fillna(0.0)

    delta_t = (t_flow - t_ret).clip(lower=0)
    # flow_l is total litres in the interval
    mass_kg = flow_l * (RHO / 1000.0)  # L → kg (1 L ≈ 1 kg for water)
    q_kj = mass_kg * CP * delta_t * hx_eff
    q_kwh = q_kj / 3600.0
    return q_kwh.clip(lower=0)
