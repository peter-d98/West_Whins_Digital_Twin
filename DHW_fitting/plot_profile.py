#!/usr/bin/env python3
"""
Quick diagnostic: plot 12 monthly DHW demand-profile histograms.

Reads DHW_fitting/output/dhw_profile.csv and writes a single 4x3 grid
of bar plots (mean_V_l vs time-of-day) to
DHW_fitting/output/plots/monthly_profiles.png.

Also prints per-month daily-volume totals and event counts to stdout.

Usage
-----
    python DHW_fitting/plot_profile.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILE_CSV = _REPO_ROOT / "DHW_fitting" / "output" / "dhw_profile.csv"
PLOT_DIR = _REPO_ROOT / "DHW_fitting" / "output" / "plots"

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def main() -> int:
    if not PROFILE_CSV.exists():
        print(f"ERROR: {PROFILE_CSV} not found. Run run_dhw_fitting.py first.",
              file=sys.stderr)
        return 1

    df = pd.read_csv(PROFILE_CSV)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine a common y-axis upper bound so the months are visually comparable.
    y_max = df["mean_V_l"].max() * 1.05

    fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    print("\nMonth | days | events | daily_L | annual_L (extrap)")
    print("------+------+--------+---------+-------------------")
    DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for i, month in enumerate(range(1, 13)):
        sub = df[df["month"] == month].sort_values("slot")
        ax = axes[i]
        # Use slot as x; relabel a few ticks to HH:MM
        ax.bar(sub["slot"].values, sub["mean_V_l"].values, width=1.0,
               color="steelblue", edgecolor="none")
        ax.set_title(f"{MONTH_NAMES[i]} (n_days={int(sub['n_days'].max())})",
                     fontsize=10)
        ax.set_ylim(0, y_max)
        ax.grid(True, alpha=0.3)
        # X-axis ticks at 0, 6, 12, 18, 24 hours → slots 0, 12, 24, 36, 47
        if i >= 9:
            ax.set_xticks([0, 12, 24, 36, 47])
            ax.set_xticklabels(["00:00", "06:00", "12:00", "18:00", "23:30"],
                               rotation=45)
            ax.set_xlabel("Time of day")
        if i % 3 == 0:
            ax.set_ylabel("Mean V_draw [L]")

        daily_L = sub["mean_V_l"].sum()
        events = int(sub["n_events"].sum())
        n_days = int(sub["n_days"].max())
        annual_L = daily_L * DAYS_IN_MONTH[i]
        print(f"  {MONTH_NAMES[i]} | {n_days:4d} | {events:6d} | "
              f"{daily_L:7.1f} | {annual_L:8.0f}")

    fig.suptitle(
        "Monthly DHW demand profile — mean draw volume per 30-min slot "
        "(West Whins, 6 flats)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = PLOT_DIR / "monthly_profiles.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
