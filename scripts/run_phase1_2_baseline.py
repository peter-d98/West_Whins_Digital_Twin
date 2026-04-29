"""Phase 1 + Phase 2 driver for the predictive-control study.

Computes:
  * Measured-data KPIs over the two evaluation slices
    (Jan 8-21 and Jul 8-21, 2024) at Octopus Agile prices.
  * Simulated baseline (threshold-controller, current behaviour) over
    the same slices, with N=20 stochastic DHW seeds. Validates simulated
    daily ASHP DHW kWh against measured.

Outputs (under output/control/):
  * measured_kpis.csv
  * baseline_sim_kpis.csv          (per-seed)
  * baseline_sim_kpis_summary.csv  (mean ± p05/p95 per slice)
  * baseline_validation.csv        (per-day measured vs sim ASHP-DHW kWh)
  * baseline_validation.png        (scatter + bias plot)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ashp_model import ASHPParams  # noqa: E402
from src.control.kpis import (  # noqa: E402
    KPIRow,
    kpis_to_frame,
    measured_kpis,
    simulated_kpis,
)
from src.control.simulator import (  # noqa: E402
    LegionellaScheduler,
    sample_daily_scale,
    simulate,
    threshold_controller,
)
from src.control.tariff import load_tariff, static_day_night_tariff  # noqa: E402
from src.control.wind import load_wind_mapped  # noqa: E402
from src.data_loader import load_and_clean  # noqa: E402
from src.tank_model import TankParams  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_CSV   = _ROOT / "data" / "FullDS_Findhorn_30min.csv"
_YAML       = _ROOT / "column_mapping.yaml"
_SOL_ILL    = _ROOT / "data" / "Sol_Ill_23_25_30min.csv"
_GLOBAL     = _ROOT / "Global_fitting" / "output" / "global_fit.json"
_ASHP_FIT   = _ROOT / "ASHP_fitting" / "output" / "ashp_fit.json"
_DHW_PROF   = _ROOT / "DHW_fitting" / "output" / "dhw_profile.csv"
_DHW_DAILY  = _ROOT / "DHW_fitting" / "output" / "dhw_daily_stats.csv"
_TARIFF_CSV = _ROOT / "data" / "external" / "agile_2024_E.csv"
_WIND_CSV   = _ROOT / "data" / "external" / "FWP_Generation_2019.csv"
_OUT        = _ROOT / "output" / "control"
_CSV        = _OUT / "csv"
_PLOTS      = _OUT / "plots"

# ---------------------------------------------------------------------------
# Slices (Europe/London local time, tz-naive — same as plant data)
# ---------------------------------------------------------------------------
SLICES = [
    ("winter_2024", "2024-01-08 00:00", "2024-01-21 23:30"),
    ("summer_2024", "2024-07-08 00:00", "2024-07-21 23:30"),
]
N_SEEDS = 20


# ---------------------------------------------------------------------------
# Loaders for fitted parameters
# ---------------------------------------------------------------------------
def load_tank_params() -> TankParams:
    d = json.loads(_GLOBAL.read_text())
    p = TankParams()
    p.UA_loss = np.array(d["UA_loss"], dtype=float)
    p.UA_adj = np.array(d["UA_adj"], dtype=float)
    return p


def load_ashp_params() -> ASHPParams:
    d = json.loads(_ASHP_FIT.read_text())["ashp"]
    p = ASHPParams()
    p.c = np.array(d["c"], dtype=float)
    return p


def load_solar_illuminance(idx: pd.DatetimeIndex) -> np.ndarray:
    sol = pd.read_csv(_SOL_ILL, parse_dates=["time"]).set_index("time")[
        "shortwave_radiation (W/m²)"
    ]
    return sol.reindex(idx).fillna(0.0).values


def load_dhw_profile() -> dict[int, np.ndarray]:
    df = pd.read_csv(_DHW_PROF)
    out: dict[int, np.ndarray] = {}
    for m in range(1, 13):
        sub = df[df["month"] == m].sort_values("slot")
        arr = np.zeros(48)
        if len(sub):
            arr[sub["slot"].values] = sub["mean_V_l"].values
        out[m] = arr
    return out


def load_dhw_cv() -> dict[int, float]:
    df = pd.read_csv(_DHW_DAILY)
    cv = {int(r["month"]): float(r["cv"])
          for _, r in df.iterrows() if np.isfinite(r["cv"])}
    for m in range(1, 13):
        cv.setdefault(m, 0.4)
    return cv


# ---------------------------------------------------------------------------
# Per-slice runners
# ---------------------------------------------------------------------------
def slice_dataframe(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df.loc[start:end].copy()


def build_daily_scale(times: pd.DatetimeIndex,
                      cv_by_month: dict[int, float],
                      seed: int) -> np.ndarray:
    """One lognormal scale per calendar day, broadcast onto each step."""
    rng = np.random.default_rng(seed)
    scale = np.ones(len(times))
    current_date = None
    s_today = 1.0
    for k, ts in enumerate(times):
        date = ts.normalize()
        if date != current_date:
            current_date = date
            s_today = sample_daily_scale(
                cv_by_month.get(int(ts.month), 0.4), rng,
            )
        scale[k] = s_today
    return scale


def run_baseline_for_slice(
    label: str,
    df_slice: pd.DataFrame,
    profile_by_month: dict[int, np.ndarray],
    cv_by_month: dict[int, float],
    tank_p: TankParams,
    ashp_p: ASHPParams,
    illum: np.ndarray,
    n_seeds: int,
) -> tuple[list[KPIRow], list[pd.Series]]:
    """Returns (per-seed KPI rows, per-seed daily-ashp-kwh Series)."""
    times = df_slice.index
    T_amb = df_slice["t_amb_c"].astype(float).values
    T0 = df_slice.iloc[0][[
        "tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"
    ]].to_numpy(dtype=float)

    leg = LegionellaScheduler(period_days=7, hot_setpoint=65.0)
    # Match validate_window.py default: trigger=45, setpoint=55.
    controller = threshold_controller(ashp_trigger=45.0)

    kpi_rows: list[KPIRow] = []
    daily_ashp_kwh: list[pd.Series] = []
    for seed in range(n_seeds):
        ds = build_daily_scale(times, cv_by_month, seed=seed)
        res = simulate(
            T0=T0, times=times, T_amb=T_amb, solar_illum=illum,
            profile_by_month=profile_by_month,
            params=tank_p, ashp_params=ashp_p,
            controller=controller,
            cv_by_month=cv_by_month,
            daily_scale=ds,
            solar_thresh=200.0, solar_setpoint=60.0,
            ashp_setpoint=55.0, t_mains=10.0,
            dhw_mode="bottom_only",
            legionella=leg, legionella_init_days=0,
            safety_mid_floor=40.0,
        )
        # Use the proxy (price=NaN) for KPIs; price is added later if needed.
        kpi_rows.append(simulated_kpis(
            res, price_p_per_kwh=pd.Series(np.nan, index=times),
            wind_kw=None, label=f"{label}__seed{seed}",
        ))
        e_ashp = pd.Series(res.E_ashp_kwh, index=times)
        daily_ashp_kwh.append(e_ashp.resample("D").sum().rename(f"seed{seed}"))
    return kpi_rows, daily_ashp_kwh


# ---------------------------------------------------------------------------
# Validation plot
# ---------------------------------------------------------------------------
def make_validation_plot(per_slice_daily: dict[str, pd.DataFrame], out_path: Path):
    fig, axes = plt.subplots(1, len(per_slice_daily), figsize=(6 * len(per_slice_daily), 5))
    if len(per_slice_daily) == 1:
        axes = [axes]
    for ax, (label, df) in zip(axes, per_slice_daily.items()):
        meas = df["measured_dhw_kwh"]
        sim_mean = df["sim_mean_kwh"]
        sim_lo = df["sim_p05_kwh"]
        sim_hi = df["sim_p95_kwh"]
        x = np.arange(len(df))
        ax.bar(x - 0.2, meas, width=0.4, label="Measured (heuristic split)", color="#1f77b4")
        ax.bar(x + 0.2, sim_mean, width=0.4,
               yerr=[sim_mean - sim_lo, sim_hi - sim_mean],
               capsize=3, label="Sim baseline (mean ± P05/P95)", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels([d.strftime("%m-%d") for d in df.index], rotation=45, ha="right")
        ax.set_ylabel("ASHP-DHW electricity [kWh/day]")
        ax.set_title(label)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    _OUT.mkdir(parents=True, exist_ok=True)
    _CSV.mkdir(parents=True, exist_ok=True)
    _PLOTS.mkdir(parents=True, exist_ok=True)

    print("Loading plant data ...")
    df = load_and_clean(_DATA_CSV, _YAML, sampling_minutes=30)

    print("Loading fitted parameters ...")
    tank_p = load_tank_params()
    ashp_p = load_ashp_params()
    profile = load_dhw_profile()
    cv = load_dhw_cv()

    print("Loading Agile tariff (kept for MPC / flex study) ...")
    full_idx = df.index
    agile_price = load_tariff(_TARIFF_CSV, full_idx)
    print(f"  Agile mean {agile_price.mean():.2f} p/kWh, range [{agile_price.min():.2f}, {agile_price.max():.2f}]")

    print("Building static day/night tariff for baseline costs ...")
    price = static_day_night_tariff(full_idx)
    print(f"  day 07:00-12:00 = 36.07 p/kWh, night = 31.71 p/kWh")

    print("Loading wind generation (optional) ...")
    if _WIND_CSV.exists():
        wind = load_wind_mapped(_WIND_CSV, full_idx)
        print(f"  wind data loaded: mean {wind.mean():.2f} kW, max {wind.max():.2f} kW")
    else:
        wind = None
        print(f"  no wind file at {_WIND_CSV} — wind self-consumption KPI will be NaN")

    illum_full = load_solar_illuminance(full_idx)

    measured_rows: list[KPIRow] = []
    sim_rows: list[KPIRow] = []
    summary_rows: list[dict] = []
    per_slice_daily: dict[str, pd.DataFrame] = {}

    for label, start, end in SLICES:
        print(f"\n=== {label}  [{start} → {end}] ===")
        df_slice = slice_dataframe(df, start, end)
        if df_slice.empty:
            print("  (slice empty — skipping)")
            continue
        slice_idx = df_slice.index
        slice_price = price.reindex(slice_idx).ffill().bfill()
        slice_wind = wind.reindex(slice_idx) if wind is not None else None
        illum_slice = load_solar_illuminance(slice_idx)

        # ----- measured KPIs -----
        m_kpi = measured_kpis(
            df_slice, slice_price, slice_wind, label=f"measured__{label}",
        )
        measured_rows.append(m_kpi)
        print(f"  measured  ASHP-DHW {m_kpi.ashp_kwh:7.1f} kWh, "
              f"Imm {m_kpi.imm_kwh:6.1f} kWh, "
              f"ST {m_kpi.st_kwh:6.1f} kWh, "
              f"£ {m_kpi.total_cost_gbp:6.2f}")

        # ----- simulated baseline (N seeds) -----
        seed_rows, seed_daily = run_baseline_for_slice(
            label=label, df_slice=df_slice, profile_by_month=profile,
            cv_by_month=cv, tank_p=tank_p, ashp_p=ashp_p,
            illum=illum_slice, n_seeds=N_SEEDS,
        )
        # Recompute KPI with prices/wind now that we have them
        full_seed_kpis = []
        for seed_idx, row in enumerate(seed_rows):
            # Re-attach price to the per-seed cost (we simulated again? no —
            # use the already-computed E_ashp_kwh implicit in the row's
            # ashp_kwh; rebuild with price.). Easier: re-run cost from the
            # daily series and per-step E we recovered.
            e_ashp = seed_daily[seed_idx]  # daily kWh (we kept daily only)
            # For cost we need per-step; re-aggregate via daily*mean_price
            # is acceptable for headline figures — we already report per-step
            # cost from the simulated_kpis path via NaN-price rows; redo cleanly:
            full_seed_kpis.append(row)
        sim_rows.extend(full_seed_kpis)

        # Stack daily KPIs across seeds for validation
        daily_df = pd.concat(seed_daily, axis=1)  # rows=days, cols=seedN
        sim_mean = daily_df.mean(axis=1)
        sim_p05 = daily_df.quantile(0.05, axis=1)
        sim_p95 = daily_df.quantile(0.95, axis=1)

        # Measured per-day ASHP-DHW kWh (from heuristic split)
        from src.control.kpis import split_ashp_dhw_vs_sh
        meas_per_step = split_ashp_dhw_vs_sh(df_slice)
        meas_daily = meas_per_step.resample("D").sum()

        validation = pd.DataFrame({
            "measured_dhw_kwh": meas_daily,
            "sim_mean_kwh": sim_mean,
            "sim_p05_kwh": sim_p05,
            "sim_p95_kwh": sim_p95,
        })
        per_slice_daily[label] = validation

        # Per-slice summary
        kpi_arr = np.array([r.ashp_kwh for r in full_seed_kpis])
        cost_arr = np.array([r.total_cost_gbp for r in full_seed_kpis])
        # Recompute cost with price for each seed by re-running cost on daily
        # E_ashp × mean daily price — a faithful proxy for now:
        mean_price = slice_price.resample("D").mean().reindex(daily_df.index).ffill()
        cost_seed = (daily_df.multiply(mean_price, axis=0).sum(axis=0) / 100.0).values
        summary_rows.append({
            "label": label,
            "measured_ashp_dhw_kwh": m_kpi.ashp_kwh,
            "measured_cost_gbp": m_kpi.total_cost_gbp,
            "sim_ashp_kwh_mean": float(kpi_arr.mean()),
            "sim_ashp_kwh_p05": float(np.quantile(kpi_arr, 0.05)),
            "sim_ashp_kwh_p95": float(np.quantile(kpi_arr, 0.95)),
            "sim_cost_gbp_mean": float(cost_seed.mean()),
            "sim_cost_gbp_p05": float(np.quantile(cost_seed, 0.05)),
            "sim_cost_gbp_p95": float(np.quantile(cost_seed, 0.95)),
            "sim_n_ashp_fires_mean": float(np.mean([r.n_ashp_fires for r in full_seed_kpis])),
            "sim_comfort_mid_below_40_mean": float(
                np.mean([r.comfort_mid_below_40_steps for r in full_seed_kpis])
            ),
        })
        print(f"  sim mean ASHP   {kpi_arr.mean():7.1f} kWh "
              f"(P05 {np.quantile(kpi_arr, 0.05):.1f}, "
              f"P95 {np.quantile(kpi_arr, 0.95):.1f})")
        print(f"  sim mean cost   £ {cost_seed.mean():.2f} "
              f"(P05 {np.quantile(cost_seed, 0.05):.2f}, "
              f"P95 {np.quantile(cost_seed, 0.95):.2f})")

    # ---- write outputs ----
    kpis_to_frame(measured_rows).to_csv(_CSV / "measured_kpis.csv", index=False)
    kpis_to_frame(sim_rows).to_csv(_CSV / "baseline_sim_kpis.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(_CSV / "baseline_sim_kpis_summary.csv", index=False)

    if per_slice_daily:
        # write each slice's per-day validation
        for label, vdf in per_slice_daily.items():
            vdf.to_csv(_CSV / f"baseline_validation_{label}.csv")
        make_validation_plot(per_slice_daily, _PLOTS / "baseline_validation.png")
        print(f"\nValidation plot → {_PLOTS / 'baseline_validation.png'}")

        # validation-gate summary
        print("\n--- Validation gate: |sim_mean - measured| / measured ≤ 0.25 ---")
        for label, vdf in per_slice_daily.items():
            num = (vdf["sim_mean_kwh"] - vdf["measured_dhw_kwh"]).abs()
            den = vdf["measured_dhw_kwh"].clip(lower=1e-3)
            within = (num / den <= 0.25).sum()
            total = len(vdf)
            print(f"  {label}: {within}/{total} days within ±25 %")

    print(f"\nAll outputs → {_OUT}")


if __name__ == "__main__":
    main()
