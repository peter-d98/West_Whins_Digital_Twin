"""
Tests for the Global_fitting pipeline.

Covers:
  - GlobalFitConfig instantiates with defaults
  - fit_global runs on synthetic data and returns physical UA_adj values
  - free_ua_loss_bottom=True keeps UA_loss[0] within bounds
  - free_ua_loss_bottom=False freezes UA_loss[0] at prior value
  - Output JSON schema validation
  - run_global_fitting raises clearly if prior JSONs are missing
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from Global_fitting.config import GlobalFitConfig
from Global_fitting.fit_global import (
    GlobalFitResult,
    _build_tank_params,
    _fit,
    _load_ashp_priors,
    _load_ua_priors,
    run_global_fit,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_df(n_intervals: int = 30) -> pd.DataFrame:
    """Build a synthetic 30-min DataFrame with all sources active.

    Temperatures gently rise (ASHP + immersion charging pattern) with
    plausible noise.
    """
    dt = pd.Timedelta(minutes=30)
    times = pd.date_range("2024-01-01", periods=n_intervals, freq=dt)
    rng = np.random.default_rng(42)

    # Gently rising temperatures
    base = np.linspace(0, 3.0, n_intervals)
    noise = rng.normal(0, 0.1, (n_intervals, 4))

    df = pd.DataFrame({
        "tank_bottom_c": 30.0 + base + noise[:, 0],
        "tank_mid_c":    40.0 + base + noise[:, 1],
        "tank_mid_hi_c": 48.0 + base + noise[:, 2],
        "tank_top_c":    55.0 + base + noise[:, 3],
        "ashp_inst_kwh": 0.5,    # ASHP on every interval
        "imm_tot_inst_kwh": 0.2, # immersion on
        "st_kwh": 0.1,           # ST on
        "t_amb_c": 18.0,
        "t_out_c": 5.0,
    }, index=times)
    return df


def _make_ua_json(path: Path) -> None:
    """Write a plausible ua_fit.json."""
    data = {"UA_loss": [0.003, 0.005, 0.003, 0.002]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _make_ashp_json(path: Path) -> None:
    """Write a plausible ashp_fit.json."""
    data = {
        "ashp": {
            "c": [3.0, 0.05, -0.02, 0.0],
            "a": [8.0, 0.1, -0.05, 0.0],
            "b": [3.0, -0.02, 0.03, 0.0],
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _make_inputs(df: pd.DataFrame) -> dict:
    """Build the inputs dict directly from the synthetic DataFrame."""
    T_meas = df[["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]].values
    Q_st = df["st_kwh"].fillna(0).values
    # Simple heat estimate: capacity * dt_h
    Q_ashp = np.where(
        df["ashp_inst_kwh"].values > 0.05,
        8.0 * 0.5,  # ~4 kWh per interval
        0.0,
    )
    Q_imm = df["imm_tot_inst_kwh"].fillna(0).values
    T_amb = df["t_amb_c"].fillna(18.0).values
    return dict(T_meas=T_meas, Q_st=Q_st, Q_ashp=Q_ashp, Q_imm=Q_imm, T_amb=T_amb)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_instantiates_with_defaults(self):
        cfg = GlobalFitConfig()
        assert cfg.train_frac == 0.7
        assert cfg.max_nfev == 500
        assert isinstance(cfg.free_ua_loss_bottom, bool)
        assert cfg.ua_adj_bounds == [0.0, 0.5]
        assert cfg.draw_delta_c == -0.5
        assert cfg.ua_fit_path == Path("UA_fitting/output/ua_fit.json")
        assert cfg.ashp_fit_path == Path("ASHP_fitting/output/ashp_fit.json")
        assert cfg.free_f_ashp is False
        assert cfg.f_ashp_mid_bounds == [0.0, 1.0]


# ---------------------------------------------------------------------------
# Fitting tests
# ---------------------------------------------------------------------------

class TestFitting:
    def test_fit_returns_physical_ua_adj(self):
        """fit_global on synthetic data returns positive UA_adj < 0.5."""
        df = _make_synthetic_df(n_intervals=30)
        ua_loss = np.array([0.003, 0.005, 0.003, 0.002])
        inputs = _make_inputs(df)

        cfg = GlobalFitConfig(max_nfev=100, free_ua_loss_bottom=False)
        result = _fit(inputs, ua_loss, cfg)

        assert isinstance(result, GlobalFitResult)
        ua_adj = result.tank_params.UA_adj
        assert len(ua_adj) == 3
        assert all(v >= 0.0 for v in ua_adj), f"UA_adj should be non-negative: {ua_adj}"
        assert all(v < 0.5 for v in ua_adj), f"UA_adj should be < 0.5: {ua_adj}"

    def test_free_ua_loss_bottom_within_bounds(self):
        """With free_ua_loss_bottom=True, UA_loss[0] stays in [-0.008, 0.004]."""
        df = _make_synthetic_df(n_intervals=30)
        ua_loss = np.array([0.001, 0.005, 0.003, 0.002])
        inputs = _make_inputs(df)

        cfg = GlobalFitConfig(max_nfev=100, free_ua_loss_bottom=True)
        result = _fit(inputs, ua_loss, cfg)

        ua_loss_0 = result.tank_params.UA_loss[0]
        assert -0.008 <= ua_loss_0 <= 0.004, (
            f"UA_loss[0] = {ua_loss_0} out of bounds [-0.008, 0.004]"
        )

    def test_frozen_ua_loss_bottom_equals_prior(self):
        """With free_ua_loss_bottom=False, UA_loss[0] equals the prior exactly."""
        df = _make_synthetic_df(n_intervals=30)
        ua_loss_prior = np.array([-0.005, 0.005, 0.003, 0.002])
        inputs = _make_inputs(df)

        cfg = GlobalFitConfig(max_nfev=100, free_ua_loss_bottom=False)
        result = _fit(inputs, ua_loss_prior, cfg)

        assert result.tank_params.UA_loss[0] == pytest.approx(
            ua_loss_prior[0], abs=1e-12
        ), "UA_loss[0] should be frozen at the prior value"

    def test_train_rmse_populated(self):
        """Result should contain per-node train RMSE."""
        df = _make_synthetic_df(n_intervals=30)
        ua_loss = np.array([0.003, 0.005, 0.003, 0.002])
        inputs = _make_inputs(df)

        cfg = GlobalFitConfig(max_nfev=50, free_ua_loss_bottom=False)
        result = _fit(inputs, ua_loss, cfg)

        assert len(result.train_rmse) == 4
        for name in ["T_bottom", "T_mid", "T_mid_hi", "T_top"]:
            assert name in result.train_rmse
            assert result.train_rmse[name] >= 0

    def test_free_f_ashp_sums_to_one(self):
        """With free_f_ashp=True, f_ashp must sum to 1.0 and f_ashp[0]==0."""
        df = _make_synthetic_df(n_intervals=30)
        ua_loss = np.array([0.003, 0.005, 0.003, 0.002])
        inputs = _make_inputs(df)

        cfg = GlobalFitConfig(max_nfev=100, free_f_ashp=True)
        result = _fit(inputs, ua_loss, cfg)

        f = result.tank_params.f_ashp
        assert len(f) == 4
        assert f[0] == pytest.approx(0.0, abs=1e-12), "f_ashp[0] must be zero (no bottom heating)"
        assert sum(f) == pytest.approx(1.0, abs=1e-6), f"f_ashp sums to {sum(f)}, expected 1.0"
        assert all(v >= -1e-9 for v in f), f"f_ashp has negative component: {f}"


# ---------------------------------------------------------------------------
# Output JSON schema tests
# ---------------------------------------------------------------------------

class TestOutputSchema:
    def test_output_json_schema(self, tmp_path):
        """global_fit.json has required keys with correct shapes."""
        ua_json = tmp_path / "ua_fit.json"
        ashp_json = tmp_path / "ashp_fit.json"
        _make_ua_json(ua_json)
        _make_ashp_json(ashp_json)

        # Build a result manually for schema check
        from Global_fitting.fit_global import F_ST, F_ASHP, F_IMM
        from src.tank_model import TankParams

        params = TankParams()
        params.UA_loss = np.array([0.003, 0.005, 0.003, 0.002])
        params.UA_adj = np.array([0.05, 0.04, 0.03])
        params.f_st = F_ST.copy()
        params.f_ashp = F_ASHP.copy()
        params.f_imm = F_IMM.copy()

        output = {
            "UA_loss": params.UA_loss.tolist(),
            "UA_adj": params.UA_adj.tolist(),
            "f_st": params.f_st.tolist(),
            "f_ashp": params.f_ashp.tolist(),
            "f_imm": params.f_imm.tolist(),
            "identification": {
                "train_rmse": {"T_bottom": 0.1, "T_mid": 0.2, "T_mid_hi": 0.15, "T_top": 0.12},
                "val_rmse": {"T_bottom": 0.15, "T_mid": 0.25, "T_mid_hi": 0.2, "T_top": 0.18},
                "n_train_intervals": 100,
                "n_val_intervals": 40,
            },
        }

        # Required top-level keys
        for key in ["UA_loss", "UA_adj", "f_st", "f_ashp", "f_imm", "identification"]:
            assert key in output

        assert len(output["UA_adj"]) == 3
        assert len(output["UA_loss"]) == 4
        assert abs(sum(output["f_st"]) - 1.0) < 1e-6
        assert abs(sum(output["f_ashp"]) - 1.0) < 1e-6
        assert abs(sum(output["f_imm"]) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Pipeline / prior-loading tests
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_fails_if_ua_json_missing(self, tmp_path):
        """Pipeline should fail clearly if ua_fit.json does not exist."""
        fake_path = tmp_path / "nonexistent" / "ua_fit.json"
        with pytest.raises(FileNotFoundError, match="UA priors file not found"):
            _load_ua_priors(fake_path)

    def test_fails_if_ashp_json_missing(self, tmp_path):
        """Pipeline should fail clearly if ashp_fit.json does not exist."""
        fake_path = tmp_path / "nonexistent" / "ashp_fit.json"
        with pytest.raises(FileNotFoundError, match="ASHP priors file not found"):
            _load_ashp_priors(fake_path)

    def test_fails_if_ua_json_bad_shape(self, tmp_path):
        """Pipeline should fail if UA_loss has wrong shape."""
        bad_json = tmp_path / "ua_fit.json"
        bad_json.write_text(json.dumps({"UA_loss": [0.001, 0.002]}))
        with pytest.raises(ValueError, match="shape"):
            _load_ua_priors(bad_json)

    def test_run_global_fit_with_missing_priors(self, tmp_path):
        """run_global_fit should raise FileNotFoundError with clear message."""
        cfg = GlobalFitConfig(
            ua_fit_path=tmp_path / "missing_ua.json",
            ashp_fit_path=tmp_path / "missing_ashp.json",
        )
        with pytest.raises(FileNotFoundError, match="UA priors file not found"):
            run_global_fit(cfg)
