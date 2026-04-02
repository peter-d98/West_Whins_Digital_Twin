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
        If True, ``UA_loss[0]`` is free within ``[-0.008, 0.004]``.
        If False, it is frozen at its prior value from ua_fit.json.
    ua_adj_bounds : List[float]
        Lower and upper bounds for each ``UA_adj`` element [kW/K].
    draw_delta_c : float
        Timesteps where the measured bottom-node temperature drops by more than
        this value (default -1.0 °C) are flagged as draw events and excluded
        from the least-squares residual.  Set to a large negative value (e.g.
        -999) to disable draw masking.
    free_f_ashp : bool
        If True, the ASHP heat-distribution fractions for mid and mid-hi are
        free parameters: ``f_ashp = [0, a, b, 1-a-b]`` with ``a, b ≥ 0`` and
        ``a + b ≤ 1``.  If False, ``f_ashp`` is frozen at ``[0, 0, 0, 1]``.
    f_ashp_mid_bounds : List[float]
        Lower and upper bounds for *a* (mid fraction) and *b* (mid-hi fraction)
        when ``free_f_ashp=True``.  Both parameters share these bounds.
    """

    ua_fit_path: Path = field(
        default_factory=lambda: Path("UA_fitting/output/ua_fit.json")
    )
    ashp_fit_path: Path = field(
        default_factory=lambda: Path("ASHP_fitting/output/ashp_fit.json")
    )
    data_csv: Path = field(
        default_factory=lambda: Path("data/FullDS_Findhorn.csv")
    )
    column_mapping_yaml: Path = field(
        default_factory=lambda: Path("column_mapping.yaml")
    )
    train_frac: float = 0.7
    max_nfev: int = 500
    free_ua_loss_bottom: bool = False
    ua_adj_bounds: List[float] = field(default_factory=lambda: [0.0, 0.5])
    draw_delta_c: float = -1.0
    free_f_ashp: bool = False
    f_ashp_mid_bounds: List[float] = field(default_factory=lambda: [0.0, 1.0])
