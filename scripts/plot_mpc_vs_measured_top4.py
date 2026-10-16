"""MPC vs measured comparison plot — simplified (top-4 temperature panels only).

Same simulation as `plot_mpc_vs_measured.py` but the figure shows only the four
tank-temperature panels (Top, Mid-Hi, Mid, Bottom), with larger text and a
trimmed title (no per-window stats line). Output filename gets a `_top4`
suffix so it sits alongside the original.

Usage:
    python scripts/plot_mpc_vs_measured_top4.py \\
        --start "2024-07-08 00:00" --end "2024-07-12 23:30" \\
        --tariff daynight
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ashp_model import ASHPParams  # noqa: E402
from src.control.mpc import (  # noqa: E402
    MPCConfig,
    MPCController,
    plan_legionella_overrides,
)
from src.control.simulator import (  # noqa: E402
    LegionellaScheduler,
    simulate,
)
from src.control.tariff import load_tariff, static_day_night_tariff  # noqa: E402
from src.data_loader import load_and_clean  # noqa: E402
from src.tank_model import TankParams  # noqa: E402

NODE_COLS = ["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]
NODE_LABELS = ["Bottom", "Mid", "Mid-Hi", "Top"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start",  default="2024-07-08 00:00")
    p.add_argument("--end",    default="2024-07-12 23:30")
    p.add_argument("--tariff", choices=["daynight", "agile"], default="daynight")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out_path = Path(args.out) if args.out else (
        _ROOT / "output" / "control" / "plots"
        / f"mpc_vs_measured_{args.start[:10]}_to_{args.end[:10]}_{args.tariff}_top4.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_and_clean(_ROOT / "data" / "FullDS_Findhorn_30min.csv",
                        _ROOT / "column_mapping.yaml", sampling_minutes=30)
    sl = df.loc[args.start:args.end].copy()
    times = sl.index

    # Params
    d = json.loads((_ROOT / "Global_fitting" / "output" / "global_fit.json").read_text())
    tp = TankParams(); tp.UA_loss = np.array(d["UA_loss"]); tp.UA_adj = np.array(d["UA_adj"])
    ad = json.loads((_ROOT / "ASHP_fitting" / "output" / "ashp_fit.json").read_text())["ashp"]
    ap = ASHPParams(); ap.c = np.array(ad["c"])

    # DHW profile + CV
    dp = pd.read_csv(_ROOT / "DHW_fitting" / "output" / "dhw_profile.csv")
    profile = {}
    for m in range(1, 13):
        sub = dp[dp.month == m].sort_values("slot")
        a = np.zeros(48)
        if len(sub):
            a[sub.slot.values] = sub.mean_V_l.values
        profile[m] = a
    dd = pd.read_csv(_ROOT / "DHW_fitting" / "output" / "dhw_daily_stats.csv")
    cv = {int(r["month"]): float(r["cv"]) for _, r in dd.iterrows()
          if np.isfinite(r["cv"])}
    for m in range(1, 13):
        cv.setdefault(m, 0.4)

    sol = pd.read_csv(_ROOT / "data" / "Sol_Ill_23_25_30min.csv",
                      parse_dates=["time"]).set_index("time")["shortwave_radiation (W/m²)"]
    illum = sol.reindex(times).fillna(0.0).values

    # Tariff
    if args.tariff == "agile":
        price_full = load_tariff(_ROOT / "data" / "external" / "agile_2024_E.csv",
                                 df.index)
        tariff_label = "Octopus Agile"
    else:
        price_full = static_day_night_tariff(df.index)
        tariff_label = "Operator day/night (07-12 = 36.07p, else 31.71p)"
    price = price_full.reindex(times).ffill().bfill()

    leg_overrides = plan_legionella_overrides(times, price)

    # MPC against this tariff
    cfg = MPCConfig(horizon_steps=96)
    mpc = MPCController(
        times=times, T_amb=sl.t_amb_c.values, solar_illum=illum,
        price_p_per_kwh=price, profile_by_month=profile, cv_by_month=cv,
        params=tp, ashp_params=ap,
        leg=LegionellaScheduler(7, 65.0), leg_overrides=leg_overrides, cfg=cfg,
    )

    T0 = sl.iloc[0][NODE_COLS].to_numpy(dtype=float)
    res = simulate(
        T0=T0, times=times, T_amb=sl.t_amb_c.values, solar_illum=illum,
        profile_by_month=profile, params=tp, ashp_params=ap,
        controller=mpc,
        cv_by_month={m: 0.0 for m in range(1, 13)},
        daily_scale=np.ones(len(times)),
        solar_thresh=200.0, solar_setpoint=60.0,
        ashp_setpoint=55.0, t_mains=10.0,
        dhw_mode="bottom_only",
        legionella=LegionellaScheduler(7, 65.0),
        legionella_overrides=leg_overrides,
        safety_mid_floor=40.0,
    )
    pred = res.T_hist[1:]
    fl_ashp = res.ashp_flags
    fl_st = res.solar_flags
    fl_imm = res.imm_flags

    meas = sl[NODE_COLS].to_numpy(dtype=float)

    # ----- Plot: 4 temperature panels only, larger fonts -----
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    })

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    dt = times[1] - times[0]

    for ax, idx in zip(axes, [3, 2, 1, 0]):
        ax.plot(times, meas[:, idx], color=colours[idx], lw=1.8,
                label=f"Measured {NODE_LABELS[idx]}")
        ax.plot(times, pred[:, idx], color=colours[idx], lw=1.4, ls="--",
                label="MPC (sim)")
        for k in np.where(fl_ashp)[0]:
            ax.axvspan(times[k], times[k] + dt, color="purple", alpha=0.15)
        for k in np.where(fl_st)[0]:
            ax.axvspan(times[k], times[k] + dt, color="gold", alpha=0.15)
        for k in np.where(fl_imm)[0]:
            ax.axvspan(times[k], times[k] + dt, color="red", alpha=0.30)
        if idx == 1:
            ax.axhline(40.0, color="grey", ls=":", lw=1)
        ax.set_ylabel(f"{NODE_LABELS[idx]} [°C]")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d-%b\n%H:%M"))

    shading_handles = [
        Patch(facecolor="purple", alpha=0.15, label="MPC ASHP fire"),
        Patch(facecolor="gold", alpha=0.15, label="MPC solar fire"),
        Patch(facecolor="red", alpha=0.30, label="MPC immersion (legionella)"),
    ]
    fig.legend(handles=shading_handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), frameon=True, fontsize=12)

    fig.suptitle(
        f"MPC vs measured  |  {args.start} → {args.end}  |  tariff: {tariff_label}",
        fontsize=15,
    )
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
