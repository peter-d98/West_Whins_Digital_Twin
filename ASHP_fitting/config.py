"""
ASHP_fitting.config – Default thresholds and configuration for ASHP-only
interval detection and performance-map fitting.

All thresholds are physically motivated defaults.  They can be overridden
from the CLI (see run_ashp_fitting.py) or by passing a custom ASHPFitConfig
object to the library functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ASHPFitConfig:
    """Central configuration for the ASHP fitting pipeline.

    Attributes
    ----------
    sampling_minutes : int
        Interval cadence of the input data [minutes].
    train_frac : float
        Fraction of the time-ordered dataset to use for fitting (0–1).
    ashp_off_kwh : float
        ASHP is considered OFF when ``ashp_inst_kwh <= ashp_off_kwh`` [kWh].
        Used to detect *non*-ASHP intervals.  Intervals where ASHP is above
        this threshold are candidates for ASHP-only windows.
    st_off_kwh : float
        Solar-thermal is OFF when ``st_kwh <= st_off_kwh`` [kWh].
    imm_off_kwh : float
        Immersion is OFF when ``imm_tot_inst_kwh <= imm_off_kwh`` [kWh].
    draw_delta_c : float
        Maximum allowed drop in ``tank_bottom_c`` between consecutive samples
        to NOT flag a draw event [°C].  Default -1.0 (a drop > 1 °C is a draw).
    min_ashp_intervals : int
        Minimum number of consecutive ASHP-only intervals to form a window.
        Default 1 — even single-interval windows are accepted.
    apply_high_load_filter : bool
        If True, only fit the power map from intervals above the 75th
        percentile of ASHP electrical energy.  Default False.
    node_cols : List[str]
        Canonical column names for the 4 tank temperature nodes (bottom→top).
    t_amb_col : str
        Column name for the plant-room ambient temperature.
    t_out_col : str
        Column name for outdoor air temperature.
    """

    # --- data cadence -------------------------------------------------------
    sampling_minutes: int = 30

    # --- train/val split ----------------------------------------------------
    train_frac: float = 0.7

    # --- ASHP-only interval detection thresholds ----------------------------
    ashp_off_kwh: float = 0.013     # kWh per interval — below = ASHP off
    st_off_kwh: float = 0.05        # kWh per interval — below = ST off
    imm_off_kwh: float = 0.01       # kWh per interval — below = immersion off

    # --- draw detection -----------------------------------------------------
    draw_delta_c: float = -1.0      # °C per interval (bottom node)

    # --- windowing ----------------------------------------------------------
    min_ashp_intervals: int = 1     # minimum consecutive ASHP-only intervals

    # --- map fitting --------------------------------------------------------
    apply_high_load_filter: bool = False

    # --- plotting / diagnostics ---------------------------------------------
    n_plot_windows: int = 5

    # --- column names (match src.data_loader canonical names) ---------------
    node_cols: List[str] = field(default_factory=lambda: [
        "tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c",
    ])
    t_amb_col: str = "t_amb_c"
    t_out_col: str = "t_out_c"
