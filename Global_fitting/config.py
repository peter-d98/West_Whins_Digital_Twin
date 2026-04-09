"""
Global_fitting.config – Configuration dataclass for the global tank-parameter
fitting pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class GlobalFitConfig:
    """Central configuration for the global fitting pipeline.

    Attributes
    ----------
    ua_fit_path : Path
        Path to ``ua_fit.json`` produced by UA_fitting.
    ashp_fit_path : Path
        Path to ``ashp_fit.json`` produced by ASHP_fitting.
    data_csv : Path
        Path to the 30-min cleaned CSV dataset.
    column_mapping_yaml : Path
        Path to column_mapping.yaml.
    train_frac : float
        Fraction of the time-ordered dataset to use for training (0–1).
    max_nfev : int
        Maximum function evaluations for ``scipy.optimize.least_squares``.
    free_ua_loss_bottom : bool
        If True (default), ``UA_loss[0]`` is free within ``[-0.008, 0.004]``,
        warm-started from its prior value in ua_fit.json.
        If False, it is frozen at its prior value.
    ua_adj_bounds : List[float]
        Lower and upper bounds for each ``UA_adj`` element [kW/K].
    draw_delta_c : float
        Timesteps where the measured bottom-node temperature drops by more than
        this value (default -1.0 °C) are flagged as draw events and excluded
        from the least-squares residual.  Set to a large negative value (e.g.
        -999) to disable draw masking.
    collapse_mid_rising_c : float
        Steps where ``tank_mid_c`` rises by more than this value [°C] between
        consecutive intervals are flagged as ASHP collapse events and excluded
        from the OSA residual.  These are the steps where the HX circulation
        pump causes sudden stratification redistribution; the model has no
        representation of this transient and the steps would otherwise bias
        ``UA_adj`` upward.  Default 2.0 °C matches ``mid_rising_c`` in
        ``ASHPFitConfig``.
    ashp_off_kwh : float
        ASHP is considered idle when ``Q_ashp < ashp_off_kwh`` [kWh].  Used
        together with ``collapse_mid_rising_c`` to identify collapse steps
        (mid rises sharply but no ASHP power is yet drawn).  Default 0.016 kWh
        matches ``ashp_off_kwh`` in ``ASHPFitConfig``.
    """

    ua_fit_path: Path = field(
        default_factory=lambda: Path("UA_fitting/output/ua_fit.json")
    )
    ashp_fit_path: Path = field(
        default_factory=lambda: Path("ASHP_fitting/output/ashp_fit.json")
    )
    data_csv: Path = field(
        default_factory=lambda: Path("data/FullDS_Findhorn_5min.csv")
    )
    column_mapping_yaml: Path = field(
        default_factory=lambda: Path("column_mapping_5min.yaml")
    )
    sampling_minutes: int = 5
    train_frac: float = 0.7
    max_nfev: int = 500
    free_ua_loss_bottom: bool = True
    ua_adj_bounds: List[float] = field(default_factory=lambda: [0.0, 0.5])
    draw_delta_c: float = -0.25
    collapse_mid_rising_c: float = 2.0
    ashp_off_kwh: float = 0.016
