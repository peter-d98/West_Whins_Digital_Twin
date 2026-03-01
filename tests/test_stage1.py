"""
Unit tests for the Stage-1 digital twin modules.
"""

from __future__ import annotations

import textwrap
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH  = ROOT / "FullDS_Findhorn.csv"
YAML_PATH = ROOT / "column_mapping.yaml"


@pytest.fixture(scope="module")
def column_cfg() -> dict:
    return yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def small_csv(tmp_path_factory) -> Path:
    """Create a tiny synthetic CSV for fast unit tests."""
    rows = textwrap.dedent("""\
    Time,ASHP Elec [kWh],Backup Imm Elec [kWh],Imm Elec [kWh],ST Flow [L],ST Flow T [°C],ST Power [kW],ST Ret T [°C],ST Tot Energy [MWh],ST Tot Vol [L],PV Inst [kW],Tank Bottom [°C],Tank Mid [°C],Tank Mid Hi [°C],Tank Top [°C],ASHP Inst [kWh],Imm Tot [kWh],Imm Tot Inst [kWh],T_amb [C]
    01/01/2024 00:00,100.0,0,0,0,30,0,30,10,100,-0.001,25,45,50,55,0.1,0,0,15
    01/01/2024 00:30,100.1,0,0,0,30,0,30,10,100,-0.002,25.5,44.8,49.8,54.9,0.1,0,0,15
    01/01/2024 01:00,100.2,0,0,0,30,0,30,10,100,0.0,26,44.5,49.5,54.7,0.1,0,0,15
    01/01/2024 01:30,100.3,0,0,0,30,2,30,10,100,0.0,26.5,44.2,49.2,54.5,0.1,0,0,15
    01/01/2024 02:00,100.4,0,0,0,30,0,30,10,100,0.0,27,44,49,54.3,0.1,0,0,15
    """)
    p = tmp_path_factory.mktemp("data") / "test.csv"
    p.write_text(rows, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# data_loader tests
# ---------------------------------------------------------------------------


class TestDataLoader:
    def test_load_column_mapping(self, column_cfg):
        from src.data_loader import load_column_mapping
        cfg = load_column_mapping(YAML_PATH)
        assert "tank" in cfg
        assert cfg["time"]["name"] == "Time"

    def test_load_and_clean_small(self, small_csv):
        from src.data_loader import load_and_clean
        df = load_and_clean(small_csv, YAML_PATH)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert "tank_top_c" in df.columns
        assert "t_amb_c" in df.columns
        assert len(df) >= 4

    def test_negative_pv_clipped(self, small_csv):
        from src.data_loader import load_and_clean
        df = load_and_clean(small_csv, YAML_PATH)
        if "pv_inst_kw" in df.columns:
            assert (df["pv_inst_kw"] >= 0).all()

    def test_node_ordering_check(self, small_csv):
        from src.data_loader import load_and_clean, node_ordering_check
        df = load_and_clean(small_csv, YAML_PATH)
        ordered = node_ordering_check(df)
        assert ordered.dtype == bool

    @pytest.mark.skipif(not CSV_PATH.exists(), reason="Full dataset not available")
    def test_load_full_csv(self):
        from src.data_loader import load_and_clean
        df = load_and_clean(CSV_PATH, YAML_PATH)
        assert len(df) > 1000
        assert "tank_top_c" in df.columns


# ---------------------------------------------------------------------------
# solar_thermal tests
# ---------------------------------------------------------------------------


class TestSolarThermal:
    def test_compute_from_power_column(self, small_csv):
        from src.data_loader import load_and_clean
        from src.solar_thermal import compute_st_energy
        df = load_and_clean(small_csv, YAML_PATH)
        q = compute_st_energy(df)
        assert len(q) == len(df)
        assert (q >= 0).all()

    def test_derive_from_flow(self):
        """When st_power_kw is absent, derive from flow × ΔT."""
        from src.solar_thermal import compute_st_energy
        idx = pd.date_range("2024-01-01", periods=4, freq="30min")
        df = pd.DataFrame({
            "st_flow_l":       [0, 100, 200, 0],
            "st_flow_temp_c":  [30, 60, 65, 30],
            "st_return_temp_c":[30, 40, 45, 30],
        }, index=idx)
        q = compute_st_energy(df)
        assert q.iloc[0] == 0.0
        assert q.iloc[1] > 0  # positive heat from flow


# ---------------------------------------------------------------------------
# ashp_model tests
# ---------------------------------------------------------------------------


class TestASHPModel:
    def test_sink_proxy(self):
        from src.ashp_model import sink_proxy
        result = sink_proxy(np.array([40.0]), np.array([50.0]))
        np.testing.assert_allclose(result, [45.0])

    def test_predict_capacity_positive(self):
        from src.ashp_model import ASHPParams, predict_capacity
        p = ASHPParams()
        q = predict_capacity(np.array([10.0]), np.array([40.0]), p)
        assert q[0] > 0

    def test_predict_power_positive(self):
        from src.ashp_model import ASHPParams, predict_power
        p = ASHPParams()
        pel = predict_power(np.array([10.0]), np.array([40.0]), p)
        assert pel[0] > 0

    def test_cop_reasonable(self):
        from src.ashp_model import ASHPParams, predict_cop
        p = ASHPParams()
        cop = predict_cop(np.array([10.0]), np.array([40.0]), p)
        assert 1.0 < cop[0] < 8.0

    def test_fit_maps(self):
        from src.ashp_model import fit_ashp_maps
        rng = np.random.default_rng(42)
        n = 200
        T_amb = rng.uniform(5, 20, n)
        T_sink = rng.uniform(35, 55, n)
        # Synthetic power: ~3 kW avg
        P_true = 2.5 + 0.01 * T_amb + 0.02 * T_sink
        P_kwh = P_true * 0.5 + rng.normal(0, 0.05, n)
        params = fit_ashp_maps(T_amb, T_sink, None, P_kwh)
        assert params.b is not None


# ---------------------------------------------------------------------------
# tank_model tests
# ---------------------------------------------------------------------------


class TestTankModel:
    def test_step_stable(self):
        from src.tank_model import TankParams, tank_step
        T = np.array([25.0, 45.0, 50.0, 55.0])
        p = TankParams()
        T_new = tank_step(T, 0.0, 0.0, 0.0, 15.0, p)
        # Should cool slightly toward ambient with no input
        assert all(5 <= t <= 95 for t in T_new)
        # Top should still be warmest (approximately)
        assert T_new[3] > T_new[0]

    def test_simulate_length(self):
        from src.tank_model import TankParams, simulate
        T0 = np.array([25.0, 45.0, 50.0, 55.0])
        N = 10
        p = TankParams()
        T_hist = simulate(
            T0,
            np.zeros(N), np.zeros(N), np.zeros(N),
            np.full(N, 15.0), p,
        )
        assert T_hist.shape == (N + 1, 4)
        np.testing.assert_array_equal(T_hist[0], T0)

    def test_heating_increases_temperature(self):
        from src.tank_model import TankParams, simulate
        T0 = np.array([25.0, 45.0, 50.0, 55.0])
        N = 10
        p = TankParams()
        # Add substantial ASHP heat
        T_hist = simulate(
            T0,
            np.zeros(N),
            np.full(N, 5.0),   # 5 kWh per step
            np.zeros(N),
            np.full(N, 15.0),
            p,
        )
        # Average temperature should rise
        assert T_hist[-1].mean() > T0.mean()

    def test_param_serialisation_roundtrip(self):
        from src.tank_model import TankParams
        p = TankParams()
        v = p.to_vector()
        p2 = TankParams.from_vector(v)
        np.testing.assert_array_equal(p.UA_loss, p2.UA_loss)
        np.testing.assert_array_equal(p.UA_adj, p2.UA_adj)
        assert p.mix_coeff == p2.mix_coeff


# ---------------------------------------------------------------------------
# evaluation tests
# ---------------------------------------------------------------------------


class TestEvaluation:
    def test_rmse_basic(self):
        from src.evaluation import rmse
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0
        assert rmse(y, y + 1.0) == pytest.approx(1.0)

    def test_node_ordering_rate(self):
        from src.evaluation import node_ordering_rate
        T = np.array([
            [20, 40, 50, 60],
            [20, 40, 50, 60],
            [60, 40, 50, 20],  # violated
        ], dtype=float)
        rate = node_ordering_rate(T)
        assert 0 < rate < 1
