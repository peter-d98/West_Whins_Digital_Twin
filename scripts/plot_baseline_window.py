"""Visual sanity-check: plot simulated baseline against measured tank
trajectory over a short user-specified window.

Usage:
    python scripts/plot_baseline_window.py \\
        --start "2024-01-17 00:00" --end "2024-01-19 23:30"
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
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ashp_model import ASHPParams  # noqa: E402
from src.control.simulator import (  # noqa: E402
    LegionellaScheduler,
    simulate,
    threshold_controller,
)
from src.data_loader import load_and_clean  # noqa: E402
from src.tank_model import TankParams  # noqa: E402

NODE_COLS = ["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]
NODE_LABELS = ["Bottom", "Mid", "Mid-Hi", "Top"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2024-01-17 00:00")
    p.add_argument("--end",   default="2024-01-19 23:30")
    p.add_argument("--out",   default=str(_ROOT / "output" / "control" / "plots" / "baseline_sim_vs_measured.png"))
    p.add_argument("--ashp-trigger", type=float, default=45.0)
    p.add_argument("--ashp-setpoint", type=float, default=55.0)
    args = p.parse_args()

    df = load_and_clean(_ROOT / "data" / "FullDS_Findhorn_30min.csv",
                        _ROOT / "column_mapping.yaml", sampling_minutes=30)
    sl = df.loc[args.start:args.end].copy()
    if sl.empty:
        raise SystemExit(f"No rows in {args.start}..{args.end}")

    d = json.loads((_ROOT / "Global_fitting" / "output" / "global_fit.json").read_text())
    tp = TankParams(); tp.UA_loss = np.array(d["UA_loss"]); tp.UA_adj = np.array(d["UA_adj"])
    ad = json.loads((_ROOT / "ASHP_fitting" / "output" / "ashp_fit.json").read_text())["ashp"]
    ap = ASHPParams(); ap.c = np.array(ad["c"])

    # DHW profile (deterministic, scale=1)
    dp = pd.read_csv(_ROOT / "DHW_fitting" / "output" / "dhw_profile.csv")
    prof = {}
    for m in range(1, 13):
        sub = dp[dp.month == m].sort_values("slot")
        a = np.zeros(48)
        if len(sub):
            a[sub.slot.values] = sub.mean_V_l.values
        prof[m] = a

    sol = pd.read_csv(_ROOT / "data" / "Sol_Ill_23_25_30min.csv",
                      parse_dates=["time"]).set_index("time")["shortwave_radiation (W/m²)"]
    illum = sol.reindex(sl.index).fillna(0.0).values

    T0 = sl.iloc[0][NODE_COLS].to_numpy(dtype=float)
    res = simulate(
        T0=T0, times=sl.index, T_amb=sl.t_amb_c.values, solar_illum=illum,
        profile_by_month=prof, params=tp, ashp_params=ap,
        controller=threshold_controller(args.ashp_trigger),
        cv_by_month={m: 0.0 for m in range(1, 13)},
        daily_scale=np.ones(len(sl)),
        solar_thresh=200.0, solar_setpoint=60.0,
        ashp_setpoint=args.ashp_setpoint, t_mains=10.0,
        dhw_mode="bottom_only",
        legionella=LegionellaScheduler(7, 65.0),
        safety_mid_floor=40.0,
    )

    times = sl.index
    pred = res.T_hist[1:]
    meas = sl[NODE_COLS].to_numpy(dtype=float)
    fl = res.ashp_flags

    fig, axes = plt.subplots(5, 1, figsize=(13, 12), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 2, 2, 1]})
    colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    # Show top→bottom for visual clarity
    for ax, idx in zip(axes[:4], [3, 2, 1, 0]):
        ax.plot(times, meas[:, idx], color=colours[idx], lw=1.5,
                label=f"Measured {NODE_LABELS[idx]}")
        ax.plot(times, pred[:, idx], color=colours[idx], lw=1.2, ls="--",
                label="Simulated")
        # Shade ASHP fires
        for k in np.where(fl)[0]:
            ax.axvspan(times[k], times[k] + (times[1] - times[0]),
                       color="purple", alpha=0.15)
        ax.set_ylabel(f"{NODE_LABELS[idx]} [°C]")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

    # Bottom panel: cumulative heat injected (sim) vs measured ASHP electricity
    ax5 = axes[4]
    cum_q = np.cumsum(res.Q_ashp_kwh)
    cum_e = np.cumsum(res.E_ashp_kwh)
    cum_meas_e = sl["ashp_inst_kwh"].fillna(0.0).cumsum().values
    ax5.plot(times, cum_q, color="purple", lw=1.5, label="Sim cumulative Q (heat to tank)")
    ax5.plot(times, cum_e, color="purple", lw=1.2, ls="--", label="Sim cumulative E (electricity)")
    ax5.plot(times, cum_meas_e, color="black", lw=1.2,
             label="Measured cumulative ASHP_inst (incl. SH)")
    ax5.set_ylabel("kWh (cumulative)")
    ax5.grid(alpha=0.3)
    ax5.legend(loc="upper left", fontsize=8)

    n_fires = int(fl.sum())
    total_q = float(res.Q_ashp_kwh.sum())
    total_e = float(res.E_ashp_kwh.sum())
    measured_e = float(sl["ashp_inst_kwh"].fillna(0).sum())
    fig.suptitle(
        f"Baseline sim vs measured  |  {args.start} → {args.end}  "
        f"|  trigger {args.ashp_trigger}, setpoint {args.ashp_setpoint}\n"
        f"Sim: {n_fires} fires, Q={total_q:.1f} kWh, E={total_e:.1f} kWh   "
        f"Measured ASHP_inst (incl. SH): {measured_e:.1f} kWh",
        fontsize=10,
    )
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"Saved → {args.out}")
    print(f"Sim:  fires={n_fires}, Q={total_q:.1f} kWh, E={total_e:.1f} kWh "
          f"(implied COP {total_q/max(total_e,1e-6):.2f})")
    print(f"Meas: ASHP_inst total = {measured_e:.1f} kWh "
          f"(includes SH; not directly comparable to sim Q/E)")


if __name__ == "__main__":
    main()
