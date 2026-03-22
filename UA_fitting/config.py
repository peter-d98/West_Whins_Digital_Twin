"""
UA_fitting.config – Default thresholds and configuration for idle-window
detection and UA fitting.

All thresholds are physically motivated defaults.  They can be overridden
from the CLI (see run_ua_fitting.py) or by passing a custom UAConfig object
to the library functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class UAConfig:
    """Central configuration for the UA fitting pipeline.

    Attributes
    ----------
    sampling_minutes : int
        Interval cadence of the input data [minutes].  Must be 30 for now.
    train_frac : float
        Fraction of the time-ordered dataset to use for training (0–1).
    ashp_off_kwh : float
        ASHP is considered OFF when ``ashp_inst_kwh <= ashp_off_kwh`` [kWh].
    st_off_kwh : float
        Solar-thermal is OFF when ``st_kwh <= st_off_kwh`` [kWh].
    imm_off_kwh : float
        Immersion is OFF when ``imm_tot_inst_kwh <= imm_off_kwh`` [kWh].
    min_idle_intervals : int
        Minimum number of consecutive idle intervals to form a window.
        Default 2 → 60 min for 30-min data.
    draw_delta_c : float
        Maximum allowed drop in ``tank_bottom_c`` between consecutive samples
        to NOT flag a draw event [°C].  Default -2.0 (a drop > 2 °C is a draw).
    min_idle_windows : int
        Minimum number of valid idle windows required.  If fewer are found,
        the script warns and does not overwrite the output JSON.
    ridge_alpha : float
        Ridge regularisation weight for the least-squares fit.  Set to 0.0
        to disable regularisation.
    summer_months : List[int]
        Month numbers used to select "summer" windows for diagnostic plots.
    n_plot_windows : int
        Number of representative idle windows to plot.
    node_cols : List[str]
        Canonical column names for the 4 tank temperature nodes (bottom→top).
    t_amb_col : str
        Column name for the plant-room ambient temperature.
    """

    # --- data cadence -------------------------------------------------------
    sampling_minutes: int = 30

    # --- train/val split ----------------------------------------------------
    train_frac: float = 0.7

    # --- idle-window detection thresholds -----------------------------------
    ashp_off_kwh: float = 0.013     # kWh per interval
    st_off_kwh: float = 0.05       # kWh per interval
    imm_off_kwh: float = 0.01      # kWh per interval
    jump_thrsh: float = 1.5        # °C per interval (any node)
    min_idle_intervals: int = 2    # minimum consecutive intervals (2 × 30 min = 60 min)

    # --- draw detection -----------------------------------------------------
    draw_delta_c: float = -1.0     # °C per interval (bottom node)

    # --- fitting ------------------------------------------------------------
    min_idle_windows: int = 20     # minimum usable windows
    ridge_alpha: float = 1e-4      # small ridge term; 0.0 disables

    # --- plotting / diagnostics ---------------------------------------------
    summer_months: List[int] = field(default_factory=lambda: [6, 7, 8])
    n_plot_windows: int = 3

    # --- column names (match src.data_loader canonical names) ---------------
    node_cols: List[str] = field(default_factory=lambda: [
        "tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c",
    ])
    t_amb_col: str = "t_amb_c"
