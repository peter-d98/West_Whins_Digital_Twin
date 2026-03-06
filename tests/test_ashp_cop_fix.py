"""
Tests for the ASHP COP fix: back-calculation of heat delivery
and ASHP performance KPIs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src import ashp_model, evaluation
from src.identification import back_calculate_ashp_heat, run_identification
from src.tank_model import NODE_CAP

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "FullDS_Findhorn.csv"
YAML_PATH = ROOT / "column_mapping.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_df(
    n_ashp_only: int = 5,
    n_imm_on: int = 3,
    n_ashp_off: int = 2,
) -> pd.DataFrame:
    """Create a synthetic DataFrame with known ASHP-only / non-ASHP intervals.

    ASHP-only rows have a deliberate temperature rise across all 4 nodes
    so the back-calculated heat is positive and plausible.
    """
    n = n_ashp_only + n_imm_on + n_ashp_off
    idx = pd.date_range("2024-01-01", periods=n + 1, freq="30min")

    rng = np.random.default_rng(42)

    # Base temperatures with a gentle upward trend for ASHP-only rows
    T_bot = np.full(n + 1, 25.0)
    T_mid = np.full(n + 1, 45.0)
    T_mh = np.full(n + 1, 50.0)
    T_top = np.full(n + 1, 55.0)

    # Add temperature rises for ASHP-only intervals (rows 1..n_ashp_only)
    for k in range(1, n_ashp_only + 1):
        rise = 0.5 + rng.uniform(0, 0.5)
        T_bot[k] = T_bot[k - 1] + rise
        T_mid[k] = T_mid[k - 1] + rise
        T_mh[k] = T_mh[k - 1] + rise
        T_top[k] = T_top[k - 1] + rise

    # Immersion rows keep constant temperatures
    start_imm = n_ashp_only + 1
    for k in range(start_imm, start_imm + n_imm_on):
        T_bot[k] = T_bot[k - 1]
        T_mid[k] = T_mid[k - 1]
        T_mh[k] = T_mh[k - 1]
        T_top[k] = T_top[k - 1]

    # ASHP-off rows keep constant temperatures
    start_off = start_imm + n_imm_on
    for k in range(start_off, n + 1):
        T_bot[k] = T_bot[k - 1]
        T_mid[k] = T_mid[k - 1]
        T_mh[k] = T_mh[k - 1]
        T_top[k] = T_top[k - 1]

    ashp_kwh = np.zeros(n + 1)
    imm_kwh = np.zeros(n + 1)
    st_kwh = np.zeros(n + 1)
    t_amb = np.full(n + 1, 15.0)
    t_out = t_amb - 10.0

    # ASHP-only rows: ASHP on, immersion off, ST zero
    for k in range(1, n_ashp_only + 1):
        ashp_kwh[k] = 2.0

    # Immersion-on rows: both ASHP and immersion on
    for k in range(start_imm, start_imm + n_imm_on):
        ashp_kwh[k] = 2.0
        imm_kwh[k] = 1.0

    # ASHP-off rows: everything off
    # (already zeros)

    return pd.DataFrame(
        {
            "tank_bottom_c": T_bot,
            "tank_mid_c": T_mid,
            "tank_mid_hi_c": T_mh,
            "tank_top_c": T_top,
            "ashp_inst_kwh": ashp_kwh,
            "imm_tot_inst_kwh": imm_kwh,
            "st_kwh": st_kwh,
            "t_amb_c": t_amb,
            "t_out_c": t_out,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Test 1 — back_calculate_ashp_heat on synthetic data
# ---------------------------------------------------------------------------


class TestBackCalculateASHPHeat:
    def test_synthetic_ashp_only(self):
        """ASHP-only intervals have finite positive heat; others are NaN."""
        df = _make_synthetic_df(n_ashp_only=60, n_imm_on=3, n_ashp_off=2)
        Q = back_calculate_ashp_heat(df)

        assert len(Q) == len(df)

        # Rows 1..60 should be ASHP-only with finite positive values
        ashp_only_slice = Q.iloc[1:61]
        assert ashp_only_slice.notna().all(), "ASHP-only rows should be finite"
        assert (ashp_only_slice > 0).all(), "ASHP-only rows should be positive"
        # Physically plausible: between 0.1 and 15 kWh per half-hour interval
        assert (ashp_only_slice >= 0.1).all() and (ashp_only_slice <= 15.0).all(), (
            f"Heat values out of plausible range: "
            f"min={ashp_only_slice.min():.3f}, max={ashp_only_slice.max():.3f}"
        )

        # Immersion-on rows (61..63) should be NaN
        assert Q.iloc[61:64].isna().all(), "Immersion-on rows should be NaN"

        # ASHP-off rows (64..65) should be NaN
        assert Q.iloc[64:66].isna().all(), "ASHP-off rows should be NaN"

        # First row always NaN (no predecessor)
        assert pd.isna(Q.iloc[0])


# ---------------------------------------------------------------------------
# Test 2 — fallback behaviour with insufficient data
# ---------------------------------------------------------------------------


class TestBackCalculateFallback:
    def test_insufficient_intervals(self, caplog):
        """With fewer than 50 ASHP-only rows, return all NaN and warn."""
        df = _make_synthetic_df(n_ashp_only=10, n_imm_on=2, n_ashp_off=2)
        with caplog.at_level(logging.WARNING):
            Q = back_calculate_ashp_heat(df)

        assert Q.isna().all(), "All values should be NaN with insufficient data"
        assert any(
            "Insufficient" in rec.message or "fallback" in rec.message
            for rec in caplog.records
        ), "Should emit a warning about insufficient data"


# ---------------------------------------------------------------------------
# Test 3 — COP is no longer constant after fix
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CSV_PATH.exists(), reason="Full dataset not available")
class TestCOPNotConstant:
    def test_cop_varies_with_conditions(self):
        """After back-calculation fix, predicted COP should vary."""
        from src.data_loader import load_and_clean

        df = load_and_clean(CSV_PATH, YAML_PATH)
        tank_cols = ["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]
        df = df.dropna(subset=tank_cols, how="all")

        id_result, _, _ = run_identification(df, train_frac=0.7, max_nfev=50)

        T_out = np.linspace(-5, 20, 50)
        T_sink = np.linspace(30, 55, 50)
        cop_values = ashp_model.predict_cop(T_out, T_sink, id_result.ashp_params)

        assert cop_values.std() > 0.05, (
            f"COP should vary with conditions; got std={cop_values.std():.4f}"
        )


# ---------------------------------------------------------------------------
# Test 4 & 5 & 6 — Integration tests: energy balance, COP APE, tank RMSE
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CSV_PATH.exists(), reason="Full dataset not available")
class TestIntegration:

    @pytest.fixture(scope="class")
    def pipeline_runs(self):
        """Run the full pipeline twice: baseline (patched) and fixed."""
        from run_stage1 import main

        # Baseline: patch back_calculate_ashp_heat to return all NaN
        with patch(
            "src.identification.back_calculate_ashp_heat",
            side_effect=lambda df, **kw: pd.Series(np.nan, index=df.index),
        ):
            baseline = main(
                csv_path=CSV_PATH,
                yaml_path=YAML_PATH,
                output_dir=Path("/tmp/ashp_test_baseline"),
                max_nfev=50,
            )

        # Fixed: run with actual back-calculation
        fixed = main(
            csv_path=CSV_PATH,
            yaml_path=YAML_PATH,
            output_dir=Path("/tmp/ashp_test_fixed"),
            max_nfev=50,
        )

        return {"baseline": baseline, "fixed": fixed}

    def test_energy_balance_improves(self, pipeline_runs):
        """Energy balance residual should be smaller with the fix."""
        baseline = pipeline_runs["baseline"]
        fixed = pipeline_runs["fixed"]
        assert abs(fixed["train"]["energy_balance_residual_kwh"]) < abs(
            baseline["train"]["energy_balance_residual_kwh"]
        )

    def test_cop_ape_improves(self, pipeline_runs):
        """Validation COP median APE should not worsen with the fix."""
        baseline = pipeline_runs["baseline"]
        fixed = pipeline_runs["fixed"]
        assert fixed["val"]["cop_errors"]["median_ape"] <= baseline["val"]["cop_errors"]["median_ape"]

    def test_tank_rmse_no_degradation(self, pipeline_runs):
        """No node RMSE should worsen by more than 10%."""
        baseline = pipeline_runs["baseline"]
        fixed = pipeline_runs["fixed"]
        for node in ["T_bottom", "T_mid", "T_mid_hi", "T_top"]:
            assert fixed["val"]["node_rmse"][node] <= baseline["val"]["node_rmse"][node] * 1.10, (
                f"{node} RMSE degraded: "
                f"fixed={fixed['val']['node_rmse'][node]:.4f}, "
                f"baseline={baseline['val']['node_rmse'][node]:.4f}"
            )


# ---------------------------------------------------------------------------
# Test 7 — ashp_performance_kpis returns plausible values
# ---------------------------------------------------------------------------


class TestASHPPerformanceKPIs:
    def test_plausible_values(self):
        """KPIs should be in physically plausible ranges."""
        n = 40
        idx = pd.date_range("2024-01-01", periods=n, freq="30min")
        rng = np.random.default_rng(99)

        # 20 ASHP-on, 20 off
        ashp_kwh = np.zeros(n)
        ashp_kwh[:20] = rng.uniform(1.0, 3.0, 20)

        df = pd.DataFrame(
            {
                "tank_mid_c": rng.uniform(40, 50, n),
                "tank_top_c": rng.uniform(50, 60, n),
                "t_out_c": rng.uniform(0, 15, n),
                "ashp_inst_kwh": ashp_kwh,
            },
            index=idx,
        )

        p = ashp_model.ASHPParams()
        kpis = evaluation.ashp_performance_kpis(df, p)

        assert 1.0 <= kpis["spf"] <= 6.0, f"SPF out of range: {kpis['spf']}"
        assert 1.0 <= kpis["mean_cop_on"] <= 6.0, f"mean_cop_on out of range: {kpis['mean_cop_on']}"
        assert 0.0 <= kpis["frac_cop_above_3"] <= 1.0
        assert 0.0 <= kpis["ashp_runtime_frac"] <= 1.0


# ---------------------------------------------------------------------------
# Test 8 — ashp_performance_kpis graceful handling of empty data
# ---------------------------------------------------------------------------


class TestASHPPerformanceKPIsEmpty:
    def test_no_ashp_on(self):
        """All NaN when no ASHP-on intervals exist."""
        n = 20
        idx = pd.date_range("2024-01-01", periods=n, freq="30min")
        df = pd.DataFrame(
            {
                "tank_mid_c": np.full(n, 45.0),
                "tank_top_c": np.full(n, 55.0),
                "t_out_c": np.full(n, 10.0),
                "ashp_inst_kwh": np.zeros(n),
            },
            index=idx,
        )

        p = ashp_model.ASHPParams()
        kpis = evaluation.ashp_performance_kpis(df, p)

        assert np.isnan(kpis["spf"])
        assert np.isnan(kpis["mean_cop_on"])
        assert np.isnan(kpis["frac_cop_above_3"])
        assert np.isnan(kpis["ashp_runtime_frac"])
