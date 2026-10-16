"""Phase 3 ablation — top-node comfort variant.

Same 4-arm structure as ``run_phase3_ablation.py`` but with the comfort
floor enforced on the **top** node (index 3) rather than the mid node
(index 1). The MPC is also re-pointed to optimise against the static
day/night tariff (the tariff actually deployed at West Whins), since
that's the relevant signal for this experiment.

Arms (all use top-node comfort):
    A_top45  : threshold(top<45)
    B_top40  : threshold(top<40)
    C_mpc40  : MPC, comfort floor 40 \u00b0C, optimises D/N
    D_mpc45  : MPC, comfort floor 45 \u00b0C, optimises D/N

Outputs go to ``output/control/topnode_ablation/`` so they're easy to
distinguish from the mid-node runs.
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

TOP_NODE = 3  # 0=bottom, 1=mid, 2=mid-hi, 3=top

# ---- separate output folders ----
_OUT   = _ROOT / "output" / "control" / "topnode_ablation"
_CSV   = _OUT / "csv"
_PLOTS = _OUT / "plots"
_CSV.mkdir(parents=True, exist_ok=True)
_PLOTS.mkdir(parents=True, exist_ok=True)

SLICES = [
    ("winter_2024", "2024-01-08 00:00", "2024-01-21 23:30"),
    ("summer_2024", "2024-07-08 00:00", "2024-07-21 23:30"),
]
N_SEEDS = 5
HORIZON_STEPS = 96

ARMS = [
    ("A_top45", "threshold", 45.0),
    ("B_top40", "threshold", 40.0),
    ("C_mpc40", "mpc",       40.0),
    ("D_mpc45", "mpc",       45.0),
]


# ---- loaders ----
def load_tank_params() -> TankParams:
    d = json.loads((_ROOT / "Global_fitting" / "output" / "global_fit.json").read_text())
    p = TankParams(); p.UA_loss = np.array(d["UA_loss"]); p.UA_adj = np.array(d["UA_adj"])
    return p


def load_ashp_params() -> ASHPParams:
    d = json.loads((_ROOT / "ASHP_fitting" / "output" / "ashp_fit.json").read_text())["ashp"]
    p = ASHPParams(); p.c = np.array(d["c"])
    return p


def load_dhw_profile():
    df = pd.read_csv(_ROOT / "DHW_fitting" / "output" / "dhw_profile.csv")
    out = {}
    for m in range(1, 13):
        sub = df[df.month == m].sort_values("slot")
        a = np.zeros(48)
        if len(sub):
            a[sub.slot.values] = sub.mean_V_l.values
        out[m] = a
    return out


def load_dhw_cv():
    df = pd.read_csv(_ROOT / "DHW_fitting" / "output" / "dhw_daily_stats.csv")
    cv = {int(r["month"]): float(r["cv"]) for _, r in df.iterrows() if np.isfinite(r["cv"])}
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


def make_controller(kind, floor, *, times, T_amb, illum, dn_slice,
                    profile, cv, tank_p, ashp_p, leg, leg_overrides):
    if kind == "threshold":
        return threshold_controller(floor, node_index=TOP_NODE)
    if kind == "mpc":
        cfg = MPCConfig(horizon_steps=HORIZON_STEPS,
                        comfort_mid_min=floor,
                        comfort_node=TOP_NODE)
        return MPCController(
            times=times, T_amb=T_amb, solar_illum=illum,
            price_p_per_kwh=dn_slice,         # MPC optimises against day/night
            profile_by_month=profile, cv_by_month=cv,
            params=tank_p, ashp_params=ashp_p,
            leg=leg, leg_overrides=leg_overrides, cfg=cfg,
        )
    raise ValueError(kind)


def run_one(controller, *, T0, times, T_amb, illum, profile, cv,
            tank_p, ashp_p, daily_scale, leg, leg_overrides, floor):
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
        safety_mid_floor=floor,
        safety_node_idx=TOP_NODE,
    )


def main():
    df = load_and_clean(_ROOT / "data" / "FullDS_Findhorn_30min.csv",
                        _ROOT / "column_mapping.yaml", sampling_minutes=30)
    sol = pd.read_csv(_ROOT / "data" / "Sol_Ill_23_25_30min.csv",
                      parse_dates=["time"]).set_index("time")[
                          "shortwave_radiation (W/m\u00b2)"]
    agile_full = load_tariff(_ROOT / "data" / "external" / "agile_2024_E.csv",
                             df.index)
    dn_full    = static_day_night_tariff(df.index)
    wind_full  = load_wind_mapped(_ROOT / "data" / "external"
                                  / "FWP_Generation_2019.csv", df.index)
    profile = load_dhw_profile()
    cv      = load_dhw_cv()
    tank_p  = load_tank_params()
    ashp_p  = load_ashp_params()

    leg = LegionellaScheduler(period_days=7, hot_setpoint=65.0)

    rows = []
    for label, t0, t1 in SLICES:
        print(f"\n=== {label}  [{t0} \u2192 {t1}] ===")
        sl = df.loc[t0:t1]
        times = sl.index
        T_amb = sl.t_amb_c.astype(float).values
        illum = sol.reindex(times).fillna(0.0).values
        T0 = sl.iloc[0][["tank_bottom_c", "tank_mid_c",
                         "tank_mid_hi_c", "tank_top_c"]].to_numpy(dtype=float)
        agile_slice = agile_full.reindex(times).ffill().bfill()
        dn_slice    = dn_full.reindex(times).ffill().bfill()
        wind_slice  = wind_full.reindex(times)

        # Plan legionella against the day/night tariff (since this is the
        # tariff the MPC is now optimising for).
        leg_overrides = plan_legionella_overrides(
            times, dn_slice, period_days=7, init_days_since_last=0,
        )
        print(f"  legionella overrides: {int(leg_overrides.sum())}")

        for seed in range(N_SEEDS):
            ds = build_daily_scale(times, cv, seed=seed)
            for arm_id, kind, floor in ARMS:
                ctrl = make_controller(
                    kind, floor,
                    times=times, T_amb=T_amb, illum=illum,
                    dn_slice=dn_slice,
                    profile=profile, cv=cv,
                    tank_p=tank_p, ashp_p=ashp_p,
                    leg=leg, leg_overrides=leg_overrides,
                )
                res = run_one(
                    ctrl, T0=T0, times=times, T_amb=T_amb, illum=illum,
                    profile=profile, cv=cv,
                    tank_p=tank_p, ashp_p=ashp_p,
                    daily_scale=ds, leg=leg, leg_overrides=leg_overrides,
                    floor=floor,
                )
                e_a = pd.Series(res.E_ashp_kwh, index=times)
                e_i = pd.Series(res.E_imm_kwh, index=times)
                cost_a = cost_gbp(e_a, agile_slice) + cost_gbp(e_i, agile_slice)
                cost_d = cost_gbp(e_a, dn_slice) + cost_gbp(e_i, dn_slice)
                wsc = wind_self_consumption_pct(e_a + e_i, wind_slice)
                q_st = float(np.asarray(res.Q_st_kwh).sum())
                q_as = float(np.asarray(res.Q_ashp_kwh).sum())
                q_im = float(np.asarray(res.Q_imm_kwh).sum())
                q_t  = q_st + q_as + q_im
                top_seq = res.T_hist[1:, TOP_NODE]
                mid_seq = res.T_hist[1:, 1]
                rows.append({
                    "label": label, "arm": arm_id,
                    "kind": kind, "floor_C": floor, "seed": seed,
                    "ashp_kwh": float(e_a.sum()),
                    "imm_kwh":  float(e_i.sum()),
                    "n_ashp_fires": int(res.ashp_flags.sum()),
                    "cost_agile_gbp":   cost_a,
                    "cost_daynight_gbp": cost_d,
                    "wind_self_consumption_pct": wsc,
                    "Q_solar_kwh": q_st, "Q_ashp_kwh": q_as, "Q_imm_kwh": q_im,
                    "solar_heat_fraction_pct":
                        100.0 * q_st / q_t if q_t > 0 else 0.0,
                    "comfort_top_below_floor_steps":
                        int((top_seq < floor - 1e-6).sum()),
                    "comfort_mid_below_40_steps":
                        int((mid_seq < 40.0 - 1e-6).sum()),
                })
            print(f"    seed {seed}: " + "  ".join(
                f"{r['arm']}=\u00a3{r['cost_daynight_gbp']:.2f}({r['ashp_kwh']:.1f}kWh,fires {r['n_ashp_fires']})"
                for r in rows[-len(ARMS):]))

    rows_df = pd.DataFrame(rows)
    rows_df.to_csv(_CSV / "ablation_kpis_topnode.csv", index=False)

    summary = (rows_df.groupby(["label", "arm"])
               .agg(ashp_kwh_mean=("ashp_kwh", "mean"),
                    cost_dn_mean=("cost_daynight_gbp", "mean"),
                    cost_dn_p05=("cost_daynight_gbp",
                                 lambda s: float(np.quantile(s, 0.05))),
                    cost_dn_p95=("cost_daynight_gbp",
                                 lambda s: float(np.quantile(s, 0.95))),
                    cost_agile_mean=("cost_agile_gbp", "mean"),
                    n_ashp_fires_mean=("n_ashp_fires", "mean"),
                    wind_self_consumption_pct_mean=(
                        "wind_self_consumption_pct", "mean"),
                    solar_heat_fraction_pct_mean=(
                        "solar_heat_fraction_pct", "mean"),
                    comfort_top_below_floor_steps_mean=(
                        "comfort_top_below_floor_steps", "mean"),
                    comfort_mid_below_40_steps_mean=(
                        "comfort_mid_below_40_steps", "mean"),
                    )
               .reset_index())
    summary.to_csv(_CSV / "ablation_kpis_topnode_summary.csv", index=False)

    print("\n=== Top-node ablation summary (mean over seeds, \u00a3 D/N) ===")
    print(summary.pivot(index="label", columns="arm",
                        values="cost_dn_mean").round(2).to_string())

    print("\n=== Top-node deltas (\u00a3 D/N vs A_top45 baseline) ===")
    base = (summary.set_index(["label", "arm"])["cost_dn_mean"]
            .unstack("arm"))
    deltas = pd.DataFrame({
        "B-A (relax floor only)":  base["B_top40"] - base["A_top45"],
        "D-A (MPC only, strict)":  base["D_mpc45"] - base["A_top45"],
        "C-B (MPC only, relaxed)": base["C_mpc40"] - base["B_top40"],
        "C-A (combined)":          base["C_mpc40"] - base["A_top45"],
    }).round(2)
    print(deltas.to_string())
    deltas.to_csv(_CSV / "ablation_deltas_topnode.csv")

    # ---- bar plot ----
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot = summary.pivot(index="label", columns="arm",
                          values="cost_dn_mean")
    arm_order = [a[0] for a in ARMS]
    pivot = pivot[arm_order]
    pivot.plot(kind="bar", ax=ax, edgecolor="black",
               color=["#4477aa", "#88ccee", "#ee6677", "#cc3311"])
    ax.set_ylabel("Mean total electricity cost on day/night [\u00a3 / 14 d]")
    ax.set_title("Ablation \u2014 top-node comfort floor (D/N tariff)")
    ax.legend(title="arm", fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(_PLOTS / "ablation_cost_bars_topnode.png",
                dpi=130, bbox_inches="tight")
    print(f"\nAll outputs \u2192 {_OUT}")


if __name__ == "__main__":
    main()
