"""
UA_fitting – Standalone tool for fitting per-node UA_loss values from idle
tank periods in the West Whins DHW digital-twin dataset.

Physics background
------------------
When the tank is idle (no ASHP, solar-thermal, or immersion heat input and no
draw events), the only thermal process is standing loss to the plant-room
ambient air.  The energy balance for each node simplifies to:

    NODE_CAP · dT_i/dt  ≈  -UA_loss_i · (T_i - T_amb)

By collecting many idle windows we can build a linear system and solve for the
four UA_loss values via least-squares regression.

Public API
----------
- detect_idle_windows(df, cfg)  — find idle tank periods
- fit_ua(windows, cfg)          — least-squares UA_loss fitting
- plot_qc(windows, ua_fit, cfg) — diagnostic plots
"""

from UA_fitting.detector import detect_idle_windows
from UA_fitting.fit_ua import fit_ua
from UA_fitting.evaluate import plot_qc

__all__ = ["detect_idle_windows", "fit_ua", "plot_qc"]
