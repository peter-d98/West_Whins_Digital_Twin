"""Two-panel cost bar chart: D/N vs Agile tariff.

Same layout as ``plot_dn_vs_flex_bars.py`` but the right panel is
the Octopus Agile (region E) tariff. Shows mid-node baseline and
top-node MPC, with a single annotation box per slice that reports
the total saving and decomposes it into "threshold change" and
"MPC" contributions.

Note: the right-panel MPC bar uses an Agile-aware controller
(``mpc_top40_agile.csv``) so that each panel's MPC is optimising
against the tariff it is being scored on.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (
    AnchoredOffsetbox, TextArea, VPacker,
)
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent

_MID_CSV = _ROOT / "output" / "control" / "csv" / "ablation_kpis.csv"
_TOPNODE_CSV = _ROOT / "output" / "control" / "topnode_ablation" / "csv" / "ablation_kpis_topnode.csv"
_MPC_AGILE_CSV = _ROOT / "output" / "control" / "topnode_ablation" / "csv" / "mpc_top40_agile.csv"

_OUT = _ROOT / "output" / "control" / "topnode_ablation" / "plots" / "mpc_cost_bars_dn_vs_agile.png"

ARMS = ["mid_baseline", "mpc_top40"]
ARM_LABELS = {
    "mid_baseline": "Baseline (mid \u2265 45 \u00b0C)",
    "mpc_top40":    "MPC (top \u2265 40 \u00b0C)",
}
ARM_COLORS = {
    "mid_baseline": "#1f77b4",
    "mpc_top40":    "#d62728",
}
SLICE_ORDER = ["summer_2024", "winter_2024"]
SLICE_LABELS = {"summer_2024": "Summer window 2024",
                "winter_2024": "Winter window 2024"}


def _means(cost_col: str, mpc_override_csv: Path | None = None) -> pd.DataFrame:
    mid = pd.read_csv(_MID_CSV)
    mid = mid[mid["arm"] == "A_threshold45"]
    mid_means = mid.groupby("label")[cost_col].mean()

    top = pd.read_csv(_TOPNODE_CSV)
    thr = top[top["arm"] == "B_top40"].groupby("label")[cost_col].mean()
    mpc = top[top["arm"] == "C_mpc40"].groupby("label")[cost_col].mean()

    if mpc_override_csv is not None:
        mpc_o = pd.read_csv(mpc_override_csv)
        mpc = mpc_o.groupby("label")[cost_col].mean()

    return pd.DataFrame({"mid_baseline": mid_means,
                         "thr_top40":    thr,
                         "mpc_top40":    mpc}).reindex(SLICE_ORDER)


def _plot_panel(ax, pivot: pd.DataFrame, title: str, ylabel: str, ymax: float):
    n_arms = len(ARMS)
    width = 0.34
    x = np.arange(len(pivot.index))

    for i, arm in enumerate(ARMS):
        offset = (i - (n_arms - 1) / 2) * width
        bars = ax.bar(x + offset, pivot[arm].values,
                      width=width, color=ARM_COLORS[arm],
                      edgecolor="black", label=ARM_LABELS[arm])
        for b, v in zip(bars, pivot[arm].values):
            ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.012,
                    f"\u00a3{v:.2f}", ha="center", va="bottom", fontsize=9)

    for gi, slice_name in enumerate(pivot.index):
        base = pivot.loc[slice_name, "mid_baseline"]
        thr  = pivot.loc[slice_name, "thr_top40"]
        mpc  = pivot.loc[slice_name, "mpc_top40"]
        d_thr = base - thr
        d_mpc = thr  - mpc
        d_tot = base - mpc
        if abs(d_tot) < 1e-9:
            pct_thr = pct_mpc = 0.0
        else:
            pct_thr = 100.0 * d_thr / d_tot
            pct_mpc = 100.0 * d_mpc / d_tot
        thr_sign = "\u2212" if d_thr >= 0 else "+"
        mpc_sign = "\u2212" if d_mpc >= 0 else "+"

        head = TextArea(f"Total saving \u2212\u00a3{d_tot:.2f}",
                        textprops=dict(size=9, weight="bold"))
        body = TextArea(
            f"Threshold change: {thr_sign}\u00a3{abs(d_thr):.2f} "
            f"({pct_thr:+.0f}%)\n"
            f"MPC: {mpc_sign}\u00a3{abs(d_mpc):.2f} ({pct_mpc:+.0f}%)",
            textprops=dict(size=8.5, multialignment="center"),
        )
        packed = VPacker(children=[head, body], align="center", pad=0, sep=4)
        bar_top = max(base, mpc)
        box = AnchoredOffsetbox(
            loc="lower center",
            child=packed,
            pad=0.4, borderpad=0,
            frameon=True,
            bbox_to_anchor=(gi, bar_top + ymax * 0.06),
            bbox_transform=ax.transData,
        )
        box.patch.set_facecolor("white")
        box.patch.set_edgecolor("grey")
        box.patch.set_alpha(0.92)
        box.patch.set_boxstyle("round,pad=0.4")
        ax.add_artist(box)

    ax.set_xticks(x)
    ax.set_xticklabels([SLICE_LABELS[s] for s in pivot.index])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, ymax)
    ax.legend(fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)


def main():
    dn    = _means("cost_daynight_gbp")
    agile = _means("cost_agile_gbp", mpc_override_csv=_MPC_AGILE_CSV)
    print("D/N (\u00a3 / 14d):"); print(dn.round(2).to_string())
    print("\nAgile (\u00a3 / 14d):"); print(agile.round(2).to_string())

    ymax = max(dn.values.max(), agile.values.max()) * 1.75

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0), sharey=True)
    _plot_panel(axes[0], dn,
                "Mean Cost (Day/Night Tariff)",
                "Cost per 14-day window  [\u00a3]", ymax)
    _plot_panel(axes[1], agile,
                "Mean Cost (Agile Tariff)",
                "Cost per 14-day window  [\u00a3]", ymax)
    fig.tight_layout()
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUT, dpi=140, bbox_inches="tight")
    print(f"\nSaved \u2192 {_OUT}")


if __name__ == "__main__":
    main()
