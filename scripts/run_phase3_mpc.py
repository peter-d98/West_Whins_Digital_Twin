"""Phase 3 driver: MPC vs threshold-baseline over the two evaluation slices.

For each slice:
  * Pre-compute legionella overrides (cheapest 30-min slot per 7-day window
    on the Octopus Agile tariff).
  * Run baseline (threshold_controller, trigger=45) and MPC against the
    *same* N stochastic DHW seeds, same hedged forecast for the planner.
  * Cost is evaluated against the Octopus Agile tariff (the price signal
    the MPC is optimising); also report cost on the operator's static
    day/night tariff for reference.

Outputs (under output/control/):
  * mpc_sim_kpis.csv            (per-slice, per-seed, per-controller)
  * mpc_sim_kpis_summary.csv    (mean ± P05/P95 by slice × controller)
  * mpc_firing_histogram.png    (hour-of-day firing distribution)
  * mpc_vs_baseline_traces.png  (4-node temperature trace for one seed)
  * mpc_daily_cost_bars.png     (per-day £ for each controller)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ashp_model import ASHPParams  # noqa: E402
from src.control.kpis import cost_gbp, wind_self_consumption_pct  # noqa: E402
from src.control.mpc import (  # noqa: E402
    MPCConfig,
    MPCController,
    plan_legionella_overrides,
)
from src.control.simulator import (  # noqa: E402
    LegionellaScheduler,
    sample_daily_scale,
    simulate,
    threshold_controller,
)
from src.control.tariff import (  # noqa: E402
    load_tariff,
    static_day_night_tariff,
)
from src.control.wind import load_wind_mapped  # noqa: E402
from src.data_loader import load_and_clean  # noqa: E402
from src.tank_model import TankParams  # noqa: E402

# ---------------------------------------------------------------------------
# Paths / config
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

SLICES = [
    ("winter_2024", "2024-01-08 00:00", "2024-01-21 23:30"),
    ("summer_2024", "2024-07-08 00:00", "2024-07-21 23:30"),
]
N_SEEDS = 5  # smaller than baseline (MPC is ~50× slower per run)
HORIZON_STEPS = 96  # 48 h


# ---------------------------------------------------------------------------
# Loaders (re-used from baseline driver)
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


def build_daily_scale(times, cv_by_month, seed):
    rng = np.random.default_rng(seed)
    scale = np.ones(len(times))
    cur, s = None, 1.0
    for k, ts in enumerate(times):
        d = ts.normalize()
        if d != cur:
            cur = d
            s = sample_daily_scale(cv_by_month.get(int(ts.month), 0.4), rng)
        scale[k] = s
    return scale


# ---------------------------------------------------------------------------
# One controller-run for one seed
# ---------------------------------------------------------------------------
def _run_one(
    controller,
    *,
    T0, times, T_amb, illum,
    profile, cv, tank_p, ashp_p,
    daily_scale, leg, leg_overrides,
):
    return simulate(
        T0=T0, times=times, T_amb=T_amb, solar_illum=illum,
        profile_by_month=profile, params=tank_p, ashp_params=ashp_p,
        controller=controller,
        cv_by_month=cv, daily_scale=daily_scale,
        solar_thresh=200.0, solar_setpoint=60.0,
        ashp_setpoint=55.0, t_mains=10.0,
        dhw_mode="bottom_only",
        legionella=leg, legionella_init_days=0,
        legionella_overrides=leg_overrides,
        safety_mid_floor=40.0,
    )


# ---------------------------------------------------------------------------
# Per-slice study
# ---------------------------------------------------------------------------
def study_slice(
    label, df_slice, agile_price, daynight_price, wind,
    profile, cv, tank_p, ashp_p,
):
    times = df_slice.index
    T_amb = df_slice["t_amb_c"].astype(float).values
    illum = load_solar_illuminance(times)
    T0 = df_slice.iloc[0][[
        "tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c",
    ]].to_numpy(dtype=float)

    leg = LegionellaScheduler(period_days=7, hot_setpoint=65.0)
    agile_slice = agile_price.reindex(times).ffill().bfill()
    dn_slice    = daynight_price.reindex(times).ffill().bfill()
    wind_slice  = wind.reindex(times) if wind is not None else None

    # Optimal legionella schedule on Agile prices (same for both controllers
    # so the £ comparison isolates the ASHP scheduling decision).
    leg_overrides = plan_legionella_overrides(
        times, agile_slice, period_days=7, init_days_since_last=0,
    )
    print(f"  legionella overrides: {int(leg_overrides.sum())} fires "
          f"at steps {list(np.where(leg_overrides)[0])}")

    cfg = MPCConfig(horizon_steps=HORIZON_STEPS)
    rows: list[dict] = []
    seed_records: dict[str, dict] = {}  # for plot artefacts

    for seed in range(N_SEEDS):
        ds = build_daily_scale(times, cv, seed=seed)

        # ---- baseline ----
        res_b = _run_one(
            threshold_controller(45.0),
            T0=T0, times=times, T_amb=T_amb, illum=illum,
            profile=profile, cv=cv, tank_p=tank_p, ashp_p=ashp_p,
            daily_scale=ds, leg=leg, leg_overrides=leg_overrides,
        )
        # ---- MPC ----
        mpc = MPCController(
            times=times, T_amb=T_amb, solar_illum=illum,
            price_p_per_kwh=agile_slice,
            profile_by_month=profile, cv_by_month=cv,
            params=tank_p, ashp_params=ashp_p,
            leg=leg, leg_overrides=leg_overrides, cfg=cfg,
        )
        res_m = _run_one(
            mpc,
            T0=T0, times=times, T_amb=T_amb, illum=illum,
            profile=profile, cv=cv, tank_p=tank_p, ashp_p=ashp_p,
            daily_scale=ds, leg=leg, leg_overrides=leg_overrides,
        )

        for tag, res in (("baseline", res_b), ("mpc", res_m)):
            e_ashp = pd.Series(res.E_ashp_kwh, index=times)
            e_imm  = pd.Series(res.E_imm_kwh, index=times)
            cost_agile = (cost_gbp(e_ashp, agile_slice)
                          + cost_gbp(e_imm, agile_slice))
            cost_dn = (cost_gbp(e_ashp, dn_slice)
                       + cost_gbp(e_imm, dn_slice))
            wsc = wind_self_consumption_pct(e_ashp + e_imm, wind_slice)
            mid_seq = res.T_hist[1:, 1]
            top_seq = res.T_hist[1:, 3]
            q_st_total   = float(np.asarray(res.Q_st_kwh).sum())
            q_ashp_total = float(np.asarray(res.Q_ashp_kwh).sum())
            q_imm_total  = float(np.asarray(res.Q_imm_kwh).sum())
            q_total = q_st_total + q_ashp_total + q_imm_total
            solar_heat_frac = (q_st_total / q_total) if q_total > 0 else 0.0
            rows.append({
                "label": label, "controller": tag, "seed": seed,
                "ashp_kwh": float(e_ashp.sum()),
                "imm_kwh":  float(e_imm.sum()),
                "n_ashp_fires": int(res.ashp_flags.sum()),
                "cost_agile_gbp": cost_agile,
                "cost_daynight_gbp": cost_dn,
                "wind_self_consumption_pct": wsc,
                "Q_solar_kwh":   q_st_total,
                "Q_ashp_kwh":    q_ashp_total,
                "Q_imm_kwh":     q_imm_total,
                "solar_heat_fraction_pct": 100.0 * solar_heat_frac,
                "comfort_mid_below_40_steps": int((mid_seq < 40.0).sum()),
                "comfort_top_below_45_steps": int((top_seq < 45.0).sum()),
            })

        # Keep one seed for plotting (seed 0)
        if seed == 0:
            seed_records["baseline"] = {
                "T_hist": res_b.T_hist, "ashp_flags": res_b.ashp_flags,
            }
            seed_records["mpc"] = {
                "T_hist": res_m.T_hist, "ashp_flags": res_m.ashp_flags,
            }
        print(f"    seed {seed:2d}  baseline £{rows[-2]['cost_agile_gbp']:6.2f}  "
              f"mpc £{rows[-1]['cost_agile_gbp']:6.2f}  "
              f"(MPC kWh {rows[-1]['ashp_kwh']:5.1f} vs base {rows[-2]['ashp_kwh']:5.1f}, "
              f"MPC fires {rows[-1]['n_ashp_fires']})")

    return rows, seed_records, agile_slice, dn_slice


# ---------------------------------------------------------------------------
# Plotters
# ---------------------------------------------------------------------------
def plot_traces(label, times, recs, agile_slice, out_path):
    fig, axes = plt.subplots(5, 1, figsize=(13, 11), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 2, 2, 1.2]})
    node_labels = ["Bottom", "Mid", "Mid-Hi", "Top"]
    colours = {"baseline": "#1f77b4", "mpc": "#d62728"}
    for ax, idx in zip(axes[:4], [3, 2, 1, 0]):
        for tag, rec in recs.items():
            ax.plot(times, rec["T_hist"][1:, idx], color=colours[tag],
                    lw=1.2, label=f"{tag.upper()} {node_labels[idx]}")
            for k in np.where(rec["ashp_flags"])[0]:
                ax.axvspan(times[k], times[k] + (times[1] - times[0]),
                           color=colours[tag], alpha=0.10)
        if idx == 1:
            ax.axhline(40.0, color="grey", ls=":", lw=1,
                       label="comfort floor 40 °C")
        ax.set_ylabel(f"{node_labels[idx]} [°C]")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=7, ncol=2)

    ax5 = axes[4]
    ax5.plot(times, agile_slice.values, color="purple", lw=1.0)
    ax5.set_ylabel("Agile p/kWh")
    ax5.grid(alpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    fig.suptitle(f"{label}: baseline vs MPC trace (seed 0)", fontsize=11)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_firing_histogram(rows_by_slice, recs_by_slice, times_by_slice, out_path):
    n = len(rows_by_slice)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.2), squeeze=False)
    for ax, label in zip(axes[0], rows_by_slice):
        recs = recs_by_slice[label]
        times = times_by_slice[label]
        for tag, colour in (("baseline", "#1f77b4"), ("mpc", "#d62728")):
            flags = recs[tag]["ashp_flags"]
            hours = times[flags].hour
            ax.hist(hours, bins=np.arange(0, 25), alpha=0.55,
                    label=f"{tag.upper()} ({int(flags.sum())} fires)",
                    color=colour, edgecolor="black", linewidth=0.4)
        ax.set_title(label)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("ASHP fires (seed 0)")
        ax.set_xticks(range(0, 25, 3))
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_daily_cost(rows_df, out_path):
    summary = (rows_df.groupby(["label", "controller"])
               .agg(cost_agile_mean=("cost_agile_gbp", "mean"),
                    cost_dn_mean=("cost_daynight_gbp", "mean"))
               .unstack("controller"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    summary["cost_agile_mean"].plot(kind="bar", ax=axes[0],
                                    color={"baseline": "#1f77b4",
                                           "mpc": "#d62728"})
    axes[0].set_title("Mean cost (Agile)")
    axes[0].set_ylabel("£ per slice")
    axes[0].grid(alpha=0.3, axis="y")
    summary["cost_dn_mean"].plot(kind="bar", ax=axes[1],
                                 color={"baseline": "#1f77b4",
                                        "mpc": "#d62728"})
    axes[1].set_title("Mean cost (operator day/night)")
    axes[1].set_ylabel("£ per slice")
    axes[1].grid(alpha=0.3, axis="y")
    for ax in axes:
        ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
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
    tank_p = load_tank_params()
    ashp_p = load_ashp_params()
    profile = load_dhw_profile()
    cv = load_dhw_cv()

    full_idx = df.index
    print("Loading Agile tariff (MPC objective) ...")
    agile_price = load_tariff(_TARIFF_CSV, full_idx)
    print(f"  Agile mean {agile_price.mean():.2f} p/kWh, "
          f"range [{agile_price.min():.2f}, {agile_price.max():.2f}]")
    daynight_price = static_day_night_tariff(full_idx)

    if _WIND_CSV.exists():
        wind = load_wind_mapped(_WIND_CSV, full_idx)
        print(f"  wind: mean {wind.mean():.1f} kW, max {wind.max():.1f} kW")
    else:
        wind = None

    all_rows = []
    recs_by_slice = {}
    times_by_slice = {}
    agile_by_slice = {}

    for label, start, end in SLICES:
        print(f"\n=== {label}  [{start} → {end}] ===")
        df_slice = df.loc[start:end].copy()
        if df_slice.empty:
            print("  (empty)")
            continue
        rows, recs, agile_slice, _dn = study_slice(
            label=label, df_slice=df_slice,
            agile_price=agile_price, daynight_price=daynight_price,
            wind=wind, profile=profile, cv=cv,
            tank_p=tank_p, ashp_p=ashp_p,
        )
        all_rows.extend(rows)
        recs_by_slice[label] = recs
        times_by_slice[label] = df_slice.index
        agile_by_slice[label] = agile_slice
        # Trace plot per slice
        plot_traces(label, df_slice.index, recs, agile_slice,
                    _PLOTS / f"mpc_traces_{label}.png")

    rows_df = pd.DataFrame(all_rows)
    rows_df.to_csv(_CSV / "mpc_sim_kpis.csv", index=False)

    summary = (rows_df.groupby(["label", "controller"])
               .agg(ashp_kwh_mean=("ashp_kwh", "mean"),
                    ashp_kwh_p05=("ashp_kwh", lambda s: float(np.quantile(s, 0.05))),
                    ashp_kwh_p95=("ashp_kwh", lambda s: float(np.quantile(s, 0.95))),
                    cost_agile_mean=("cost_agile_gbp", "mean"),
                    cost_agile_p05=("cost_agile_gbp", lambda s: float(np.quantile(s, 0.05))),
                    cost_agile_p95=("cost_agile_gbp", lambda s: float(np.quantile(s, 0.95))),
                    cost_dn_mean=("cost_daynight_gbp", "mean"),
                    n_ashp_fires_mean=("n_ashp_fires", "mean"),
                    wind_self_consumption_pct_mean=(
                        "wind_self_consumption_pct", "mean"),
                    solar_heat_fraction_pct_mean=(
                        "solar_heat_fraction_pct", "mean"),
                    Q_solar_kwh_mean=("Q_solar_kwh", "mean"),
                    Q_ashp_kwh_mean=("Q_ashp_kwh", "mean"),
                    comfort_mid_below_40_mean=(
                        "comfort_mid_below_40_steps", "mean"),
                    )
               .reset_index())
    summary.to_csv(_CSV / "mpc_sim_kpis_summary.csv", index=False)

    print("\n=== Summary (mean over seeds) ===")
    for _, r in summary.iterrows():
        print(f"  {r['label']:>14s}  {r['controller']:>8s}  "
              f"ASHP {r['ashp_kwh_mean']:5.1f} kWh   "
              f"£Agile {r['cost_agile_mean']:5.2f} "
              f"(P05 {r['cost_agile_p05']:.2f}, P95 {r['cost_agile_p95']:.2f})   "
              f"£D/N {r['cost_dn_mean']:5.2f}   "
              f"fires {r['n_ashp_fires_mean']:.1f}   "
              f"wind% {r['wind_self_consumption_pct_mean']:.1f}   "
              f"solar% {r['solar_heat_fraction_pct_mean']:.1f}   "
              f"mid<40 steps {r['comfort_mid_below_40_mean']:.1f}")

    # --- aggregate plots ---
    plot_firing_histogram(
        rows_by_slice={l: None for l in recs_by_slice},
        recs_by_slice=recs_by_slice, times_by_slice=times_by_slice,
        out_path=_PLOTS / "mpc_firing_histogram.png",
    )
    plot_daily_cost(rows_df, _PLOTS / "mpc_cost_bars.png")

    print(f"\nAll outputs → {_OUT}")


if __name__ == "__main__":
    main()
