"""
DHW_fitting.config — Configuration for DHW draw-event detection and
monthly demand-profile construction.

Demand is inferred from drops in the tank bottom-node temperature
(the only thermal signature of a draw event in the absence of a flow
meter).  All thresholds are physically motivated defaults and may be
overridden from the CLI in run_dhw_fitting.py.
"""

from __future__ import annotations

from dataclasses import dataclass


# Bottom-node volume (litres) — must match src/tank_model.py NODE_VOLS[0].
# Defined here as a module constant so detector.py can reference it without
# importing the tank model (keeps DHW_fitting self-contained at detection time).
V_BOTTOM_L: float = 170.0

# Minimum allowed (T_bottom - T_mains) when back-calculating draw volume.
# Below this gap the volume back-calculation becomes numerically unstable
# (small denominators amplify temperature noise into huge spurious volumes).
MIN_TEMP_GAP_C: float = 2.0

# Maximum single-step draw volume (litres).  Equal to the smallest node
# volume in the tank (mid/mid_hi/top = 126.67 L); larger draws would
# require a multi-step cascade which is out of scope for the 30-min profiler.
V_DRAW_MAX_L: float = 126.67


@dataclass
class DhwFitConfig:
    """Central configuration for the DHW demand-profile pipeline.

    Attributes
    ----------
    sampling_minutes : int
        Interval cadence of the input data [minutes].  Must match the CSV.
    train_frac : float
        Fraction of the time-ordered dataset used for profile learning (0–1).
    draw_delta_c : float
        Threshold on dT_bottom per interval [°C].  A step is flagged as
        a draw event when dT_bottom < draw_delta_c.  Default -0.25 °C
        matches Global_fitting/config.py.
    t_mains_c : float
        Assumed mains cold-water inlet temperature [°C].  Fixed year-round.
    v_draw_max_l : float
        Upper clip on back-calculated single-step draw volume [L].
    min_temp_gap_c : float
        Minimum (T_bottom_prev - T_mains) required to back-calculate a
        volume [°C]; events below this gap are discarded.
    output_dir : str
        Directory for the demand profile CSV.
    diagnostics_dir : str
        Directory for raw detected events and other diagnostics.
    """

    # --- data cadence -------------------------------------------------------
    sampling_minutes: int = 30

    # --- train/val split ----------------------------------------------------
    train_frac: float = 0.7

    # --- draw detection -----------------------------------------------------
    draw_delta_c: float = -0.25      # °C per interval (bottom node)

    # --- back-calculation ---------------------------------------------------
    t_mains_c: float = 10.0          # assumed mains inlet temperature [°C]
    v_draw_max_l: float = V_DRAW_MAX_L
    min_temp_gap_c: float = MIN_TEMP_GAP_C

    # --- profile aggregation -----------------------------------------------
    # Suggestion (A): zero out (month, slot) profile entries where draws
    # occur on fewer than this fraction of days.  Filters out sparse noise
    # (e.g. one-off thermal artefacts misclassified as draws).
    min_event_frequency: float = 0.05

    # Suggestion (B): subtract a per-month baseline bottom-node rise rate
    # (from conduction / mixing during quiet intervals) before applying the
    # draw threshold.  This makes the effective detection floor lower for
    # months when the bottom node naturally warms quickly.
    use_baseline_subtraction: bool = True
    # Baseline is the median dT_bottom over rows where dT_bottom is in this
    # quiet band (excludes both draws and large charging-driven jumps).
    baseline_quiet_band_c: tuple = (-0.1, 0.5)

    # Use the entire dataset for profile learning (instead of train_frac).
    # Recommended for demand profiling because the profile is an exogenous
    # boundary condition, not a fitted dynamics parameter.
    use_all_data: bool = False

    # --- I/O ---------------------------------------------------------------
    output_dir: str = "DHW_fitting/output"
    diagnostics_dir: str = "DHW_fitting/diagnostics"
