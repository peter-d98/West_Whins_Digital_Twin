"""
ST_fitting.config – Default thresholds and configuration for ST-only
interval detection and f_st estimation.

All thresholds are physically motivated defaults.  They can be overridden
from the CLI (see run_st_fitting.py) or by passing a custom STFitConfig
object to the library functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class STFitConfig:
    """Central configuration for the solar-thermal fitting pipeline.

    Attributes
    ----------
    sampling_minutes : int
        Interval cadence of the input data [minutes].
    train_frac : float
        Fraction of the time-ordered dataset to use for fitting (0–1).
    st_flow_dt_min_c : float
        ST flow temperature must exceed tank bottom temperature by at least
        this value [°C] for an interval to be accepted.  Gate 1.
    st_flow_min_l : float
        Minimum ST flow volume per interval [L].  Gate 2.
    st_power_min_kw : float
        Minimum ST power per interval [kW].  Gate 3.
    ashp_off_kwh : float
        ASHP is OFF when ``ashp_inst_kwh <= ashp_off_kwh`` [kWh].  Gate 5.
    imm_off_kwh : float
        Immersion is OFF when ``imm_tot_inst_kwh <= imm_off_kwh`` [kWh].  Gate 6.
    min_st_intervals : int
        Minimum number of consecutive ST-only intervals to form a window.
        Default 4 (4 × 5 min = 20 min).
    node_cols : List[str]
        Canonical column names for the 4 tank temperature nodes (bottom→top).
    t_amb_col : str
        Column name for the plant-room ambient temperature.
    t_out_col : str
        Column name for outdoor air temperature.
    n_plot_windows : int
        How many windows to select for diagnostic plotting.
    gti_min_wm2 : float
        GTI threshold below which the ST system is modelled as inactive [W/m²].
    t_bottom_max_c : float
        Bottom-node temperature above which the ST system is modelled as
        inactive (saturation limit) [°C].
    """

    # --- data cadence -------------------------------------------------------
    sampling_minutes: int = 5

    # --- train/val split ----------------------------------------------------
    train_frac: float = 0.7

    # --- ST-only interval detection thresholds ------------------------------
    st_flow_dt_min_c: float = 4.0   # Gate 1: ST flow T − bottom T > this [°C]
    st_flow_min_l: float = 0.04     # Gate 2: ST flow volume > this [L/interval]
    st_power_min_kw: float = 0.1    # Gate 3: ST power > this [kW]
    # Gate 4 (bottom rising) has no threshold — requires bottom.diff() > 0
    ashp_off_kwh: float = 0.016     # Gate 5: ASHP energy ≤ this [kWh/interval]
    imm_off_kwh: float = 0.001      # Gate 6: immersion energy ≤ this [kWh/interval]

    # --- windowing ----------------------------------------------------------
    min_st_intervals: int = 4      # minimum consecutive ST-only intervals

    # --- ST model activation / saturation thresholds ------------------------
    gti_min_wm2: float = 180.0     # Minimum GTI for ST activation [W/m²]
    t_bottom_max_c: float = 55.0   # Bottom-node saturation temperature [°C]

    # --- plotting / diagnostics ---------------------------------------------
    n_plot_windows: int = 1

    # --- column names (match src.data_loader canonical names) ---------------
    node_cols: List[str] = field(default_factory=lambda: [
        "tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c",
    ])
    t_amb_col: str = "t_amb_c"
    t_out_col: str = "t_out_c"
    st_flow_temp_col: str = "st_flow_temp_c"   # ST flow temperature
    st_return_temp_col: str = "st_return_temp_c"  # ST return temperature
    st_flow_col: str = "st_flow_l"             # ST flow volume [L/interval]
    st_power_col: str = "st_power_kw"          # ST power [kW]
