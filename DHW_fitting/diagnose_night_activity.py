#!/usr/bin/env python3
"""
DHW_fitting/diagnose_night_activity.py
--------------------------------------
Diagnostic to characterise night-time / low-activity slots in the fitted
DHW profile.  Read-only: prints summaries + writes two PNG heatmaps and
one histogram into ``DHW_fitting/diagnostics/``.

Outputs
-------
1. month×slot heatmap of detection frequency (n_events / n_days).
2. month×slot heatmap of mean draw volume (litres).
3. histogram of consecutive-zero-event stretch lengths per day, with
   stratification by season.
4. console summary suggesting a frequency threshold and likely "quiet
   window" hours.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_loader import load_and_clean
from DHW_fitting.config import DhwFitConfig
from DHW_fitting.detector import detect_draw_events

PROFILE_CSV = _ROOT / "DHW_fitting" / "output" / "dhw_profile.csv"
OUT_DIR = _ROOT / "DHW_fitting" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
def heatmap(matrix: np.ndarray, title: str, cbar_label: str, out_path: Path,
            cmap: str = "viridis", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   origin="lower", extent=[0, 48, 0.5, 12.5])
    ax.set_xticks(np.arange(0, 49, 4))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)])
    ax.set_yticks(range(1, 13))
    ax.set_yticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_xlabel("Time of day (slot)")
    ax.set_ylabel("Month")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
def main():
    print("=== DHW night-inactivity diagnostic ===\n")

    # 1. Load the existing profile (frequency / mean_V_l per month×slot).
    print(f"Loading profile from {PROFILE_CSV} ...")
    p = pd.read_csv(PROFILE_CSV)

    freq_mat = np.zeros((12, 48))
    vol_mat = np.zeros((12, 48))
    for _, r in p.iterrows():
        freq_mat[int(r["month"]) - 1, int(r["slot"])] = r["frequency"]
        vol_mat[int(r["month"]) - 1, int(r["slot"])] = r["mean_V_l"]

    heatmap(freq_mat,
            "Detection frequency (n_events / n_days) per month × slot",
            "frequency", OUT_DIR / "diag_frequency_heatmap.png",
            cmap="viridis", vmin=0, vmax=freq_mat.max())
    heatmap(vol_mat,
            "Mean draw volume per month × slot [L]",
            "mean V_draw [L]", OUT_DIR / "diag_volume_heatmap.png",
            cmap="magma", vmin=0, vmax=np.percentile(vol_mat, 98))

    # 2. Frequency thresholds — show how many slots get zeroed at each.
    print("\nFrequency-floor sensitivity:")
    print(f"  {'threshold':>10} | {'slots zeroed':>14} | "
          f"{'volume removed':>16} | {'remaining annual L':>20}")
    total_annual_l = (vol_mat.sum(axis=1) * 365.25 / 12.0).sum()
    for thr in [0.05, 0.075, 0.10, 0.125, 0.15, 0.20]:
        mask = (freq_mat > 0) & (freq_mat < thr)
        n_zero = int(mask.sum())
        vol_removed = float((vol_mat * mask).sum() * 365.25 / 12.0)
        remaining = total_annual_l - vol_removed
        print(f"  {thr:>10.3f} | {n_zero:>14d} | "
              f"{vol_removed:>14.0f} L | {remaining:>18.0f} L")

    # 3. Per-hour averages: detection frequency vs mean volume.
    print("\nMean detection frequency by hour-of-day (averaged across months):")
    by_hour_freq = freq_mat.reshape(12, 24, 2).mean(axis=(0, 2))
    by_hour_vol = vol_mat.reshape(12, 24, 2).mean(axis=(0, 2))
    print(f"  {'hr':>3}  {'freq':>6}  {'meanV':>7}")
    for h in range(24):
        marker = "  <-- night" if 0 <= h < 6 else ""
        print(f"  {h:>3d}  {by_hour_freq[h]:>6.3f}  "
              f"{by_hour_vol[h]:>5.2f} L{marker}")

    # 4. Consecutive-zero-event stretches — needs the event detector.
    print("\nRunning detector on full dataset to map consecutive-zero stretches ...")
    cfg = DhwFitConfig(use_all_data=True)
    df = load_and_clean(_ROOT / "data" / "FullDS_Findhorn_30min.csv",
                        _ROOT / "column_mapping.yaml", sampling_minutes=30)
    events = detect_draw_events(df, cfg)

    # Slot-level boolean: does this slot have an event?
    ev_idx = pd.DatetimeIndex(events["timestamp"]).floor("30min")
    has_event = pd.Series(False, index=df.index)
    has_event.loc[has_event.index.intersection(ev_idx)] = True

    # Group by date, find longest consecutive zero-stretch per day.
    df_diag = pd.DataFrame({"event": has_event.astype(int)},
                            index=has_event.index)
    df_diag["date"] = df_diag.index.date
    df_diag["season"] = ((df_diag.index.month % 12) // 3).map(
        {0: "Winter", 1: "Spring", 2: "Summer", 3: "Autumn"})

    longest_by_day = []
    season_by_day = []
    for (d, sea), grp in df_diag.groupby(["date", "season"], sort=False):
        # Find longest run of zeros in slot order.
        events_arr = grp["event"].values
        max_run = cur = 0
        for v in events_arr:
            if v == 0:
                cur += 1
                if cur > max_run:
                    max_run = cur
            else:
                cur = 0
        longest_by_day.append(max_run)
        season_by_day.append(sea)
    longest_by_day = np.array(longest_by_day)
    season_by_day = np.array(season_by_day)
    # In hours (each slot = 0.5 h).
    hours = longest_by_day * 0.5

    print("\nDistribution of longest daily zero-event stretch (hours):")
    print(f"  Overall: median={np.median(hours):.1f} h, "
          f"p25={np.percentile(hours, 25):.1f} h, "
          f"p75={np.percentile(hours, 75):.1f} h, "
          f"max={hours.max():.1f} h")
    for sea in ["Winter", "Spring", "Summer", "Autumn"]:
        s = hours[season_by_day == sea]
        if len(s) == 0:
            continue
        print(f"  {sea:<7}: median={np.median(s):.1f} h, "
              f"p25={np.percentile(s, 25):.1f} h, "
              f"p75={np.percentile(s, 75):.1f} h  (n={len(s)} days)")

    # Plot histogram by season.
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.arange(0, 24.5, 0.5)
    for sea, col in zip(["Winter", "Spring", "Summer", "Autumn"],
                        ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"]):
        s = hours[season_by_day == sea]
        if len(s) == 0:
            continue
        ax.hist(s, bins=bins, alpha=0.45, color=col, label=sea, density=True)
    ax.set_xlabel("Longest daily zero-event stretch [hours]")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of longest daily DHW-event-free stretch, by season")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = OUT_DIR / "diag_zero_stretch_hist.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")

    # 5. Modal start hour of the quiet window.
    print("\nFinding modal start-hour of the longest daily quiet window ...")
    starts = []
    for (d, _), grp in df_diag.groupby(["date", "season"], sort=False):
        events_arr = grp["event"].values
        max_run = cur = 0
        max_start = cur_start = 0
        for i, v in enumerate(events_arr):
            if v == 0:
                if cur == 0:
                    cur_start = i
                cur += 1
                if cur > max_run:
                    max_run = cur
                    max_start = cur_start
            else:
                cur = 0
        if max_run >= 2 and len(grp) >= 48:
            # convert slot to hour-of-day
            start_hour = grp.index[max_start].hour + grp.index[max_start].minute / 60.0
            starts.append(start_hour)
    starts = np.array(starts)
    print(f"  Median start hour: {np.median(starts):.2f}")
    print(f"  P25 / P75: {np.percentile(starts, 25):.2f} / "
          f"{np.percentile(starts, 75):.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
