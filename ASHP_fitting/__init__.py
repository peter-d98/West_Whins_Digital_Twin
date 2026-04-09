"""
ASHP_fitting – Standalone ASHP performance-map identification from 30-minute
data using back-calculated heat delivery.

Physics background
------------------
When only the ASHP is running (no immersion, negligible solar-thermal, no
draw events), the tank energy balance can be rearranged to back-calculate
the condenser heat delivered:

    Q_ashp = NODE_CAP × Σ dT_i  +  Σ UA_loss_i × (T_i − T_amb) × dt_s

The resulting (Q_ashp, P_elec) pairs together with operating temperatures
(T_out, T_sink) are used to fit bilinear ASHP capacity and power maps.

Public API
----------
- detect_ashp_windows(df, cfg)  — find clean ASHP-only intervals
- fit_ashp(df, windows, ua_loss, cfg)  — back-calculate + map fitting
- evaluate_ashp_fit(...)        — diagnostic plots + QC
"""

from ASHP_fitting.detector import detect_ashp_windows
from ASHP_fitting.fit_ashp_maps import fit_ashp
from ASHP_fitting.evaluate import evaluate_ashp_fit

__all__ = ["detect_ashp_windows", "fit_ashp", "evaluate_ashp_fit"]
