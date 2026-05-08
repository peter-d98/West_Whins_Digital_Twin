#!/usr/bin/env python3
"""
scripts/validate_window_top4.py
-------------------------------
Compact variant of ``scripts/validate_window.py``.

Renders the same free-forward tank simulation, but the output PNG only
contains the four temperature panels (Top, Mid-Hi, Mid, Bottom).  The
solar-illuminance and DHW-draw panels are dropped, the title is reduced
to a single line ("Free forward simulation — <start> → <end>"), the
solar / ASHP shading is documented in a small dedicated legend, and the
per-panel RMSE / bias annotations use a larger, more legible font.

The plotting logic is the only thing that differs from
``validate_window.py``; all simulation, loaders and CLI behaviour are
imported and reused.  Output filename gets a ``_top4`` suffix to avoid
clobbering the full 6-panel plot.

Usage mirrors the original script:

    python scripts/validate_window_top4.py \\
        --window "2024-07-15 00:00" "2024-07-16 23:30" \\
        --dhw-mode bottom_only --n-realisations 20
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

# Make the project root importable so we can reuse validate_window.py
# without needing a scripts/__init__.py.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

import validate_window as vw  # noqa: E402  (path tweak above)
from validate_window import (  # noqa: E402
    NODE_COLS, NODE_COLOURS, NODE_LABELS,
    build_spans, main as _orig_main,
)


# ===========================================================================
# Custom plotter — top 4 temperature panels only
# ===========================================================================
def plot_window_top4(
    win: pd.DataFrame,
    realisations: list[np.ndarray],
    T_central: np.ndarray,
    solar_flags: np.ndarray,
    ashp_flags: np.ndarray,
    V_draws_med: np.ndarray,         # accepted for signature parity, unused
    solar_illum: np.ndarray,         # accepted for signature parity, unused
    out_path: Path,
    *,
    solar_thresh: float, solar_setpoint: float,
    ashp_trigger: float, ashp_setpoint: float,
    n_realisations: int,
    dhw_mode: str,
):
    """Render only the four temperature panels.

    * Title: single line — ``Free forward simulation — <start> → <end>``.
    * Solar / ASHP fires shown as orange / purple shaded bars; a small
      figure-level legend documents what each colour means.
    * RMSE / bias text in each panel uses a larger font for legibility.
    """
    # Append a top4 marker so we don't overwrite the 6-panel output.
    out_path = out_path.with_name(out_path.stem + "_top4" + out_path.suffix)

    T_meas = win[NODE_COLS].values
    times = win.index
    dt = times[1] - times[0]

    if len(realisations) > 1:
        stacked = np.stack(realisations, axis=0)
        pred_p05 = np.percentile(stacked, 5, axis=0)[1:]
        pred_p95 = np.percentile(stacked, 95, axis=0)[1:]
    else:
        pred_p05 = pred_p95 = None
    pred_central = T_central[1:]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    # Top-down physical order: Top, Mid-Hi, Mid, Bottom.
    temp_axes = {3: axes[0], 2: axes[1], 1: axes[2], 0: axes[3]}

    fig.suptitle(
        f"Free forward simulation \u2014 {times[0]:%Y-%m-%d %H:%M} \u2192 "
        f"{times[-1]:%Y-%m-%d %H:%M}",
        fontsize=14,
    )

    solar_spans = build_spans(solar_flags, times, dt)
    ashp_spans  = build_spans(ashp_flags,  times, dt)

    for i in range(4):
        ax = temp_axes[i]
        for s, e in solar_spans:
            ax.axvspan(s, e, color="orange", alpha=0.15, zorder=0)
        for s, e in ashp_spans:
            ax.axvspan(s, e, color="purple", alpha=0.10, zorder=0)

        ax.plot(times, T_meas[:, i],
                "o-", color=NODE_COLOURS[i], lw=1.4, ms=3.5,
                label="Measured", zorder=3)
        ax.plot(times, pred_central[:, i],
                "--", color=NODE_COLOURS[i], lw=1.4, alpha=0.85,
                label="Predicted", zorder=3)
        if pred_p05 is not None:
            ax.fill_between(times, pred_p05[:, i], pred_p95[:, i],
                            color=NODE_COLOURS[i], alpha=0.15,
                            label="P05\u2013P95 (stochastic)", zorder=1)

        err = pred_central[:-1, i] - T_meas[1:, i]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        bias = float(np.mean(err))
        ax.set_ylabel(f"{NODE_LABELS[i]} [\u00b0C]", fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
        ax.legend(
            title=f"RMSE={rmse:.2f}\u00b0C   bias={bias:+.2f}\u00b0C",
            loc="best", fontsize=11, title_fontsize=12,
        )
        ax.grid(True, alpha=0.3)

    # Figure-level legend documenting the shading colours.
    shade_handles = [
        Patch(facecolor="orange", alpha=0.15,
              label=f"Solar fire (illum \u2265 {solar_thresh:.0f} W/m\u00b2)"),
        Patch(facecolor="purple", alpha=0.10,
              label=f"ASHP fire (mid \u2264 {ashp_trigger:.1f}\u00b0C)"),
    ]
    fig.legend(
        handles=shade_handles,
        loc="lower center", ncol=2,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=12, frameon=True,
    )

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
    axes[-1].set_xlabel("Time", fontsize=12)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved \u2192 {out_path}")


# ===========================================================================
# Patch and run
# ===========================================================================
def main():
    # Swap in the trimmed plotter; CLI / loaders / simulator are unchanged.
    vw.plot_window = plot_window_top4
    _orig_main()


if __name__ == "__main__":
    main()
