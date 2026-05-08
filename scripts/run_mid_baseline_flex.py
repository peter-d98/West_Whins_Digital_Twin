"""One-arm helper: run the mid-node baseline (threshold mid<45) on the
Findhorn flex tariff for the same summer/winter 2024 windows.

Used solely to provide the missing baseline column for the
D/N-vs-flex MPC comparison chart. Outputs a small CSV.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ashp_model import ASHPParams  # noqa: E402
from src.control.kpis import cost_gbp  # noqa: E402
from src.control.mpc import plan_legionella_overrides  # noqa: E402
from src.control.simulator import (  # noqa: E402
    LegionellaScheduler,
    sample_daily_scale,
    simulate,
    threshold_controller,
)
from src.control.tariff import (  # noqa: E402
    load_flex_tariff,
    static_day_night_tariff,
)
from src.data_loader import load_and_clean  # noqa: E402
from src.tank_model import TankParams  # noqa: E402

_OUT_CSV = _ROOT / "output" / "control" / "topnode_flex" / "csv" / "mid_baseline_flex.csv"
SLICES = [
    ("winter_2024", "2024-01-08 00:00", "2024-01-21 23:30"),
    ("summer_2024", "2024-07-08 00:00", "2024-07-21 23:30"),
]
N_SEEDS = 5


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


def main():
    df = load_and_clean(_ROOT / "data" / "FullDS_Findhorn_30min.csv",
                        _ROOT / "column_mapping.yaml", sampling_minutes=30)
    sol = pd.read_csv(_ROOT / "data" / "Sol_Ill_23_25_30min.csv",
                      parse_dates=["time"]).set_index("time")[
                          "shortwave_radiation (W/m\u00b2)"]
    flex_full = load_flex_tariff(
        _ROOT / "data" / "external" / "Findhorn_flex_tariff.csv", df.index)
    dn_full = static_day_night_tariff(df.index)
    profile = load_dhw_profile(); cv = load_dhw_cv()
    tank_p  = load_tank_params(); ashp_p = load_ashp_params()
    leg = LegionellaScheduler(period_days=7, hot_setpoint=65.0)

    rows = []
    for label, t0, t1 in SLICES:
        sl = df.loc[t0:t1]
        times = sl.index
        T_amb = sl.t_amb_c.astype(float).values
        illum = sol.reindex(times).fillna(0.0).values
        T0 = sl.iloc[0][["tank_bottom_c", "tank_mid_c",
                         "tank_mid_hi_c", "tank_top_c"]].to_numpy(dtype=float)
        flex_slice = flex_full.reindex(times).ffill().bfill()
        dn_slice   = dn_full.reindex(times).ffill().bfill()
        leg_overrides = plan_legionella_overrides(
            times, flex_slice, period_days=7, init_days_since_last=0)
        for seed in range(N_SEEDS):
            ds = build_daily_scale(times, cv, seed=seed)
            ctrl = threshold_controller(45.0, node_index=1)  # MID NODE
            res = simulate(
                T0=T0, times=times, T_amb=T_amb, solar_illum=illum,
                profile_by_month=profile, params=tank_p, ashp_params=ashp_p,
                controller=ctrl, cv_by_month=cv, daily_scale=ds,
                solar_thresh=200.0, solar_setpoint=60.0,
                ashp_setpoint=55.0, t_mains=10.0,
                dhw_mode="bottom_only",
                legionella=leg, legionella_init_days=0,
                legionella_overrides=leg_overrides,
                safety_mid_floor=45.0, safety_node_idx=1,
            )
            e_a = pd.Series(res.E_ashp_kwh, index=times)
            e_i = pd.Series(res.E_imm_kwh,  index=times)
            rows.append({
                "label": label, "arm": "MID_baseline",
                "seed": seed,
                "ashp_kwh": float(e_a.sum()),
                "n_ashp_fires": int(res.ashp_flags.sum()),
                "cost_flex_gbp":     cost_gbp(e_a, flex_slice) + cost_gbp(e_i, flex_slice),
                "cost_daynight_gbp": cost_gbp(e_a, dn_slice)   + cost_gbp(e_i, dn_slice),
            })
            r = rows[-1]
            print(f"  {label} seed{seed}: \u00a3flex={r['cost_flex_gbp']:.2f}  "
                  f"\u00a3dn={r['cost_daynight_gbp']:.2f}  "
                  f"({r['ashp_kwh']:.1f}kWh, fires {r['n_ashp_fires']})")

    out = pd.DataFrame(rows)
    _OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(_OUT_CSV, index=False)
    print("\nMeans:")
    print(out.groupby("label")[["cost_flex_gbp", "cost_daynight_gbp"]]
          .mean().round(2).to_string())
    print(f"\nSaved \u2192 {_OUT_CSV}")


if __name__ == "__main__":
    main()
