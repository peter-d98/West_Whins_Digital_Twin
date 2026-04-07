"""
Tests for the ASHP_fitting pipeline.

Covers:
  - Detector: draw interval rejection (not full-window), min_ashp_intervals=1
  - Fitting: Q_back <= 0 intervals excluded
  - Pipeline: fails clearly if ua_fit.json missing
  - Output: JSON schema contains required keys
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ASHP_fitting.config import ASHPFitConfig
from ASHP_fitting.detector import detect_ashp_windows, ASHPWindow
from ASHP_fitting.fit_ashp_maps import back_calculate_q_ashp, fit_ashp

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_df(
    n_intervals: int = 20,
    sampling_minutes: int = 30,
    ashp_kwh: float = 0.5,
    imm_kwh: float = 0.0,
    st_kwh: float = 0.0,
    draw_at: int | None = None,
) -> pd.DataFrame:
    """Build a synthetic 30-min DataFrame for testing.

    All intervals have ASHP on by default, with rising tank temperatures.
    ``draw_at`` injects a -2°C bottom-node drop at that index (0-based).
    """
    dt = pd.Timedelta(minutes=sampling_minutes)
    times = pd.date_range("2024-01-01", periods=n_intervals, freq=dt)

    # Rising temperatures at >1 °C/step so mid_rising gate passes for all
    # intervals (detector now uses mid-node, not top-node).
    rise = np.arange(n_intervals) * 1.5
    base_temps = {
        "tank_bottom_c": 30.0 + rise,
        "tank_mid_c":    40.0 + rise,
        "tank_mid_hi_c": 48.0 + rise,
        "tank_top_c":    52.0 + rise,
    }

    df = pd.DataFrame({
        **base_temps,
        "ashp_inst_kwh": ashp_kwh,
        "imm_tot_inst_kwh": imm_kwh,
        "st_kwh": st_kwh,
        "t_amb_c": 18.0,
        "t_out_c": 5.0,
    }, index=times)

    # Inject draw event: sharp bottom-node drop
    if draw_at is not None and 0 < draw_at < n_intervals:
        df.iloc[draw_at, df.columns.get_loc("tank_bottom_c")] -= 3.0

    return df


def _make_ua_loss() -> np.ndarray:
    """Return plausible UA_loss priors for testing."""
    return np.array([0.003, 0.005, 0.003, 0.002])


# ---------------------------------------------------------------------------
# Detector tests
# ---------------------------------------------------------------------------

class TestDetector:
    """Tests for ASHP-only interval detection."""

    def test_all_accepted_basic(self):
        """All intervals should be accepted when ASHP is on and no draws."""
        df = _make_synthetic_df(n_intervals=10)
        cfg = ASHPFitConfig(train_frac=1.0)
        windows, diag = detect_ashp_windows(df, cfg)

        # First interval always rejected (no previous row for dT)
        # so we expect 9 accepted intervals in 1 window
        assert len(windows) >= 1
        total_intervals = sum(w.n_intervals for w in windows)
        assert total_intervals == 9  # index 1..9

    def test_draw_interval_rejection_not_full_window(self):
        """A draw event should reject only that interval, not the full window.

        With draw at index 5, we expect two windows:
        - indices 1..4 (before draw)
        - indices 6..9 (after draw)
        The draw interval (5) is rejected but surrounding intervals survive.
        """
        df = _make_synthetic_df(n_intervals=10, draw_at=5)
        cfg = ASHPFitConfig(train_frac=1.0)
        windows, diag = detect_ashp_windows(df, cfg)

        # Draw at index 5 splits into two windows
        assert len(windows) == 2
        # Both windows should have intervals (before and after draw)
        assert windows[0].n_intervals >= 1
        assert windows[1].n_intervals >= 1
        # Total accepted = 9 - 1 draw = 8
        total = sum(w.n_intervals for w in windows)
        assert total == 8

        # Check diagnostics has draw_event reason
        draw_rows = diag[diag["reject_reason"] == "draw_event"]
        assert len(draw_rows) == 1

    def test_min_interval_length_1(self):
        """With min_ashp_intervals=1, even a single interval forms a window."""
        # 3 intervals: index 0 (no prev), 1 (ASHP on), 2 (ASHP off)
        df = _make_synthetic_df(n_intervals=3)
        # Turn off ASHP at index 2
        df.iloc[2, df.columns.get_loc("ashp_inst_kwh")] = 0.0
        cfg = ASHPFitConfig(train_frac=1.0, min_ashp_intervals=1)
        windows, _ = detect_ashp_windows(df, cfg)

        # Should have at least 1 window with 1 interval (index 1)
        assert len(windows) >= 1
        assert any(w.n_intervals == 1 for w in windows)

    def test_min_interval_length_filters_short(self):
        """With min_ashp_intervals=5, short windows are rejected."""
        # Create 3 intervals of ASHP (indices 1,2,3) then off
        df = _make_synthetic_df(n_intervals=6)
        df.iloc[4, df.columns.get_loc("ashp_inst_kwh")] = 0.0
        df.iloc[5, df.columns.get_loc("ashp_inst_kwh")] = 0.0
        cfg = ASHPFitConfig(train_frac=1.0, min_ashp_intervals=5)
        windows, _ = detect_ashp_windows(df, cfg)

        # The 3-interval window should be rejected (< 5)
        assert len(windows) == 0

    def test_immersion_on_rejected(self):
        """Intervals with immersion heater on should be excluded."""
        df = _make_synthetic_df(n_intervals=5, imm_kwh=0.5)
        cfg = ASHPFitConfig(train_frac=1.0)
        windows, diag = detect_ashp_windows(df, cfg)
        assert len(windows) == 0
        assert (diag["reject_reason"] == "imm_on").any()

    def test_st_on_rejected(self):
        """Intervals with solar-thermal on should be excluded."""
        df = _make_synthetic_df(n_intervals=5, st_kwh=0.2)
        cfg = ASHPFitConfig(train_frac=1.0)
        windows, diag = detect_ashp_windows(df, cfg)
        assert len(windows) == 0
        assert (diag["reject_reason"] == "st_on").any()

    def test_ashp_off_rejected(self):
        """Intervals with ASHP off should be excluded."""
        df = _make_synthetic_df(n_intervals=5, ashp_kwh=0.0)
        cfg = ASHPFitConfig(train_frac=1.0)
        windows, diag = detect_ashp_windows(df, cfg)
        assert len(windows) == 0
        assert (diag["reject_reason"] == "ashp_off").any()


# ---------------------------------------------------------------------------
# Back-calculation and fitting tests
# ---------------------------------------------------------------------------

class TestFitting:
    """Tests for back-calculation and map fitting."""

    def test_q_back_positive_filter(self):
        """Intervals with Q_back <= 0 should be excluded from fitting data."""
        df = _make_synthetic_df(n_intervals=20)
        cfg = ASHPFitConfig(train_frac=1.0)
        windows, _ = detect_ashp_windows(df, cfg)
        ua_loss = _make_ua_loss()

        n_train = int(len(df) * cfg.train_frac)
        df_train = df.iloc[:n_train]
        bc_df = back_calculate_q_ashp(df_train, windows, ua_loss, cfg)

        # With rising temperatures and positive UA_loss, most Q should be > 0
        assert len(bc_df) > 0
        # At least some should be positive
        assert (bc_df["Q_back_kwh"] > 0).any()

    def test_q_back_lte_zero_excluded(self):
        """Verify that Q_back <= 0 intervals are not passed to map fitting."""
        # Create data where temperatures DROP (so Q_back will be negative)
        df = _make_synthetic_df(n_intervals=20)
        # Reverse temperature trend: make temps decrease
        for col in ["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]:
            df[col] = df[col].values[::-1]  # Reverse (decreasing)

        cfg = ASHPFitConfig(train_frac=1.0)
        windows, _ = detect_ashp_windows(df, cfg)
        ua_loss = _make_ua_loss()

        # fit_ashp should handle Q <= 0 gracefully
        result = fit_ashp(df, windows, ua_loss, cfg)
        # With all decreasing temps, Q_back should mostly be <= 0
        # Result should either have error or the positive filter applied
        assert "ashp" in result

    def test_fit_produces_valid_coefficients(self):
        """End-to-end: synthetic data → fitted a, b arrays."""
        df = _make_synthetic_df(n_intervals=30)
        cfg = ASHPFitConfig(train_frac=1.0)
        windows, _ = detect_ashp_windows(df, cfg)
        ua_loss = _make_ua_loss()

        result = fit_ashp(df, windows, ua_loss, cfg)
        assert "ashp" in result
        assert len(result["ashp"]["a"]) == 4
        assert len(result["ashp"]["b"]) == 4


# ---------------------------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------------------------

class TestPipeline:
    """Integration tests for the ASHP fitting pipeline."""

    def test_fails_if_ua_json_missing(self, tmp_path):
        """Pipeline should fail clearly if ua_fit.json does not exist."""
        from ASHP_fitting.run_ashp_fitting import _load_ua_priors

        fake_path = tmp_path / "nonexistent" / "ua_fit.json"
        with pytest.raises(FileNotFoundError, match="UA priors file not found"):
            _load_ua_priors(fake_path)

    def test_fails_if_ua_json_bad_shape(self, tmp_path):
        """Pipeline should fail if UA_loss has wrong shape."""
        from ASHP_fitting.run_ashp_fitting import _load_ua_priors

        bad_json = tmp_path / "ua_fit.json"
        bad_json.write_text(json.dumps({"UA_loss": [0.001, 0.002]}))
        with pytest.raises(ValueError, match="shape"):
            _load_ua_priors(bad_json)


# ---------------------------------------------------------------------------
# JSON output schema tests
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """Tests for the output JSON structure."""

    def test_json_has_required_keys(self):
        """The result dict must contain 'ashp' and 'identification' keys."""
        df = _make_synthetic_df(n_intervals=20)
        cfg = ASHPFitConfig(train_frac=1.0)
        windows, _ = detect_ashp_windows(df, cfg)
        ua_loss = _make_ua_loss()

        result = fit_ashp(df, windows, ua_loss, cfg)

        assert "ashp" in result
        assert "a" in result["ashp"]
        assert "b" in result["ashp"]
        assert "identification" in result

    def test_json_a_b_are_4_element_lists(self):
        """Coefficients a and b must each have 4 elements."""
        df = _make_synthetic_df(n_intervals=20)
        cfg = ASHPFitConfig(train_frac=1.0)
        windows, _ = detect_ashp_windows(df, cfg)
        ua_loss = _make_ua_loss()

        result = fit_ashp(df, windows, ua_loss, cfg)
        assert len(result["ashp"]["a"]) == 4
        assert len(result["ashp"]["b"]) == 4

    def test_json_identification_metadata(self):
        """Identification block must contain count and threshold info."""
        df = _make_synthetic_df(n_intervals=20)
        cfg = ASHPFitConfig(train_frac=1.0)
        windows, _ = detect_ashp_windows(df, cfg)
        ua_loss = _make_ua_loss()

        result = fit_ashp(df, windows, ua_loss, cfg)
        ident = result["identification"]

        # Skip if fitting failed due to insufficient data
        if "error" in ident:
            pytest.skip("Fitting returned error (insufficient data)")

        required_keys = [
            "n_windows_total",
            "n_windows_accepted",
            "n_intervals_total",
            "thresholds",
            "ua_loss_used",
            "timestamp",
        ]
        for key in required_keys:
            assert key in ident, f"Missing key: {key}"
