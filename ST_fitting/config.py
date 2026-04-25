"""
ST_fitting.config – Configuration for ST-only window detection and fitting.

Designed for 30-minute plant data.  All thresholds are physically motivated
defaults that can be overridden from the CLI (see run_st_fitting.py) or by
passing a custom STFitConfig to library functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class STFitConfig:
    """Central configuration for the solar-thermal fitting pipeline.

    Window detection rules (30-minute cadence):
      START:  ST power > st_power_start_kw  AND
              bottom-node rise > bottom_rise_start_c  AND
              ASHP off  AND  immersion off.
      CONTINUE:  bottom node still rising (dT > 0).
      END:    bottom node stopped rising for ``bottom_flat_close``
              consecutive intervals.
    """

    # --- data cadence -------------------------------------------------------
    sampling_minutes: int = 30

    # --- train/val split ----------------------------------------------------
    train_frac: float = 0.7

    # --- window start thresholds --------------------------------------------
    st_power_start_kw: float = 0.5    # ST power must exceed this to open [kW]
    bottom_rise_start_c: float = 2.0  # bottom node must rise by more than this to open [°C]
    ashp_off_kwh: float = 0.016       # ASHP is OFF when energy ≤ this [kWh/interval]
    imm_off_kwh: float = 0.001        # immersion is OFF when energy ≤ this [kWh/interval]

    # --- window continuation / closing --------------------------------------
    bottom_flat_close: int = 2        # close after this many consecutive non-rising intervals
    min_st_intervals: int = 2         # minimum intervals per valid window (2 × 30 min = 1 h)

    # --- plotting / diagnostics ---------------------------------------------
    n_plot_windows: int = 3

    # --- column names (match src.data_loader canonical names) ---------------
    node_cols: List[str] = field(default_factory=lambda: [
        "tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c",
    ])
    t_amb_col: str = "t_amb_c"
    t_out_col: str = "t_out_c"
    st_power_col: str = "st_power_kw"
