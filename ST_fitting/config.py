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
    st_on_kwh : float
        Solar-thermal is considered ON when ``st_kwh > st_on_kwh`` [kWh].
    ashp_off_kwh : float
        ASHP is OFF when ``ashp_inst_kwh <= ashp_off_kwh`` [kWh].
    imm_off_kwh : float
        Immersion is OFF when ``imm_tot_inst_kwh <= imm_off_kwh`` [kWh].
    draw_delta_c : float
        Maximum allowed drop in ``tank_bottom_c`` between consecutive samples
        to NOT flag a draw event [°C].  Default −0.25 (a drop > 0.25 °C is a
        draw).
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
    """

    # --- data cadence -------------------------------------------------------
    sampling_minutes: int = 5

    # --- train/val split ----------------------------------------------------
    train_frac: float = 0.7

    # --- ST-only interval detection thresholds ------------------------------
    st_on_kwh: float = 0.01       # kWh per interval — above = ST on
    ashp_off_kwh: float = 0.016    # kWh per interval — below = ASHP off
    imm_off_kwh: float = 0.001     # kWh per interval — below = immersion off

    # --- draw detection -----------------------------------------------------
    draw_delta_c: float = -0.1    # °C per interval (bottom node)

    # --- windowing ----------------------------------------------------------
    min_st_intervals: int = 4      # minimum consecutive ST-only intervals

    # --- plotting / diagnostics ---------------------------------------------
    n_plot_windows: int = 1

    # --- column names (match src.data_loader canonical names) ---------------
    node_cols: List[str] = field(default_factory=lambda: [
        "tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c",
    ])
    t_amb_col: str = "t_amb_c"
    t_out_col: str = "t_out_c"
