"""
Plot Agile price trace vs MPC ASHP firing schedule over a 48-hour window.

Produces two vertically stacked panels:
  (a) Octopus Agile half-hourly price (p/kWh) with negative-price shading
  (b) ASHP firing schedule (on/off bars) with mid-node temperature overlaid

Usage:
    python scripts/plot_mpc_price_vs_firing.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ashp_model import ASHPParams
from src.control.mpc import MPCConfig, MPCController, plan_legionella_overrides
from src.control.simulator import LegionellaScheduler, simulate, sample_daily_scale
from src.control.tariff import load_tariff
from src.data_loader import load_and_clean
from src.tank_model import TankParams

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
_OUT        = _ROOT / "output" / "control" / "plots"

# 48-hour winter window with interesting price variation
SLICE_START = "2024-01-15 00:00"
SLICE_END   = "2024-01-16 23:30"

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_tank_params() -> TankParams:
    d = json.loads(_GLOBAL.read_text())
    p = TankParams()
    p.UA_loss = np.array(d["UA_loss"], dtype=float)
    p.UA_adj  = np.array(d["UA_adj"],  dtype=float)
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
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    df_full = load_and_clean(_DATA_CSV, _YAML)
    df_slice = df_full.loc[SLICE_START:SLICE_END].copy()
    times = df_slice.index

    tank_p  = load_tank_params()
    ashp_p  = load_ashp_params()
    profile = load_dhw_profile()
    cv      = load_dhw_cv()

    T_amb = df_slice["t_amb_c"].astype(float).values
    illum = load_solar_illuminance(times)
    T0    = df_slice.iloc[0][
        ["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]
    ].to_numpy(dtype=float)

    agile_price = load_tariff(_TARIFF_CSV, times)

    leg = LegionellaScheduler(period_days=7, hot_setpoint=65.0)
    leg_overrides = plan_legionella_overrides(
        times, agile_price, period_days=7, init_days_since_last=0,
    )

    # Fixed daily scale (seed=0, deterministic)
    rng = np.random.default_rng(0)
    daily_scale = np.ones(len(times))
    cur, s = None, 1.0
    for k, ts in enumerate(times):
        d = ts.normalize()
        if d != cur:
            cur = d
            s = sample_daily_scale(cv.get(int(ts.month), 0.4), rng)
        daily_scale[k] = s

    print("Running MPC simulation...")
    cfg = MPCConfig(horizon_steps=96, comfort_node=3)
    mpc_ctrl = MPCController(
        times=times,
        T_amb=T_amb,
        solar_illum=illum,
        price_p_per_kwh=agile_price,
        profile_by_month=profile,
        cv_by_month=cv,
        params=tank_p,
        ashp_params=ashp_p,
        leg=leg,
        leg_overrides=leg_overrides,
        cfg=cfg,
    )

    res = simulate(
        T0=T0, times=times, T_amb=T_amb, solar_illum=illum,
        profile_by_month=profile, params=tank_p, ashp_params=ashp_p,
        controller=mpc_ctrl,
        cv_by_month=cv, daily_scale=daily_scale,
        solar_thresh=200.0, solar_setpoint=60.0,
        ashp_setpoint=55.0, t_mains=10.0,
        dhw_mode="bottom_only",
        legionella=leg, legionella_init_days=0,
        legionella_overrides=leg_overrides,
        safety_mid_floor=40.0,
    )

    print("Plotting...")
    prices = agile_price.values
    top_temp = res.T_hist[1:, 3]   # top node temperatures (post-step)

    # -----------------------------------------------------------------------
    # Figure: single dual-axis panel
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))

    # Shade ASHP-fired and legionella slots
    for k in range(len(times)):
        if res.ashp_flags[k]:
            ax.axvspan(
                times[k],
                times[k] + pd.Timedelta("30min"),
                alpha=0.55, color="#FF9800", linewidth=0,
            )
        if res.imm_flags[k]:
            ax.axvspan(
                times[k],
                times[k] + pd.Timedelta("30min"),
                alpha=0.55, color="#9C27B0", linewidth=0,
            )

    # Right axis: Agile price (solid, prominent)
    ax_r = ax.twinx()
    ax_r.step(times, prices, where="post", color="#E05C2A", linewidth=2.0,
              label="Agile price (p/kWh)", zorder=4)
    # shade negative-price intervals
    neg_mask = prices < 0
    for k in range(len(times)):
        if neg_mask[k]:
            ax_r.axvspan(
                times[k],
                times[k] + pd.Timedelta("30min"),
                alpha=0.18, color="#2196F3", linewidth=0,
            )
    ax_r.axhline(0, color="#E05C2A", linewidth=0.6, linestyle="--", alpha=0.5)
    ax_r.set_ylabel("Agile price (p/kWh)", fontsize=11, color="#E05C2A")
    ax_r.tick_params(axis="y", labelcolor="#E05C2A", labelsize=9)

    # Left axis: top-node temperature
    ax.plot(times, top_temp, color="#1565C0", linewidth=2.0,
            label="Top-node temp (°C)", zorder=3)
    ax.axhline(40.0, color="#1565C0", linewidth=1.2, linestyle=":",
               alpha=0.8, label="Comfort floor (40 °C)")
    ax.set_ylabel("Top-node temperature (°C)", fontsize=11, color="#1565C0")
    ax.tick_params(axis="y", labelcolor="#1565C0", labelsize=9)
    ax.set_ylim(30, 70)

    ax.set_title(
        f"MPC ASHP firing schedule vs Octopus Agile price\n"
        f"{pd.Timestamp(SLICE_START).strftime('%d %b %Y')} – "
        f"{pd.Timestamp(SLICE_END).strftime('%d %b %Y')}",
        fontsize=12,
    )

    ashp_patch = mpatches.Patch(color="#FF9800", alpha=0.65, label="ASHP firing")
    imm_patch  = mpatches.Patch(color="#9C27B0", alpha=0.65, label="Legionella (immersion)")
    neg_patch  = mpatches.Patch(color="#2196F3", alpha=0.35, label="Negative price")
    ax.legend(
        handles=[ax.lines[0], ax.lines[1], ashp_patch, imm_patch,
                 ax_r.lines[0], neg_patch],
        loc="upper left", fontsize=9,
    )
    ax.grid(axis="y", alpha=0.3)

    # x-axis formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=9)

    fig.tight_layout()
    out_path = _OUT / "mpc_price_vs_firing.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")

    # Print a quick summary
    print(f"\nASHP fires: {int(res.ashp_flags.sum())}  "
          f"Legionella fires: {int(res.imm_flags.sum())}")
    print(f"Total ASHP electricity: {res.E_ashp_kwh.sum():.2f} kWh")
    print(f"Total cost (Agile): {(res.E_ashp_kwh * prices / 100).sum():.2f} £")


if __name__ == "__main__":
    main()
