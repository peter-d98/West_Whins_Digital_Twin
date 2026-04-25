#!/usr/bin/env python3
"""
scripts/validate_simple_solar.py
---------------------------------
Free-forward simulation using the simplified solar and ASHP activation rules:

  Solar rule  : when solar illuminance (shortwave, W/m²) >= SOLAR_THRESH,
                all four tank nodes are set to SOLAR_SETPOINT_C at the END
                of that single 30-min timestep.  Solar input ceases once the
                set-point is reached; idle UA losses resume immediately after.

  ASHP rule   : when the predicted mid-node temperature falls to or below
                ASHP_TRIGGER_C, the upper three nodes (mid, mid-hi, top) are
                all set to ASHP_SETPOINT_C at the END of that timestep.  The
                bottom node is unaffected (ASHP HX is in mid-to-top section).

  Idle physics: between activation events the tank evolves purely according
                to the fitted UA_loss and UA_adj coefficients from
                global_fit.json.  No Q_sol or Q_ashp term is computed;
                external heat sources are represented only by the override
                rules above.

Output
------
  scripts/validate_simple_solar_<YYYYMMDD_HHMM>.png

Usage
-----
    python scripts/validate_simple_solar.py \
        --start "2024-05-10 00:00" --end "2024-05-10 23:30"

    # Adjust thresholds on the command line:
    python scripts/validate_simple_solar.py \
        --start "2024-06-02 00:00" --end "2024-06-02 23:30" \
        --solar-thresh 200 --solar-setpoint 62 \
        --ashp-trigger 45 --ashp-setpoint 55
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Step 0 – make sure the repository root is on sys.path so src.* imports work
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_loader import load_and_clean
from src.tank_model import TankParams, NODE_CAP, tank_step

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Paths (all relative to repository root)
# ---------------------------------------------------------------------------
_GLOBAL_JSON  = _ROOT / "Global_fitting" / "output" / "global_fit.json"
_DEFAULT_CSV  = _ROOT / "data" / "FullDS_Findhorn_30min.csv"
_DEFAULT_YAML = _ROOT / "column_mapping.yaml"
_SOLAR_ILL    = _ROOT / "data" / "Sol_Ill_23_25_30min.csv"
_PLOT_OUT     = _ROOT / "scripts"

# ---------------------------------------------------------------------------
# Display names / colours – shared by plot helpers
# ---------------------------------------------------------------------------
NODE_LABELS  = ["Bottom", "Mid", "Mid-Hi", "Top"]
NODE_COLS    = ["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]
NODE_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

DT_S = 1800.0   # 30-minute timestep in seconds


# ===========================================================================
# Step 1 – load and validate parameters from global_fit.json
# ===========================================================================
def load_tank_params(path: Path) -> TankParams:
    """Read UA_loss, UA_adj (and the stored distribution fractions) from the
    global fit JSON.  Only UA_loss and UA_adj are used in this simplified
    script – the f_* fractions are not needed because heat is applied as a
    direct temperature override."""
    with open(path) as fh:
        d = json.load(fh)
    p = TankParams()
    p.UA_loss = np.array(d["UA_loss"], dtype=float)
    p.UA_adj  = np.array(d["UA_adj"],  dtype=float)
    # Keep f_* at their defaults; they are not used in the simplified rules.
    return p


# ===========================================================================
# Step 2 – load solar illuminance and align to the main dataframe index
# ===========================================================================
def load_solar_illuminance(path: Path, main_index: pd.DatetimeIndex) -> np.ndarray:
    """Read Sol_Ill_23_25_30min.csv and reindex to *main_index*.
    Missing values are zero-filled (conservative: no activation on gaps)."""
    sol = pd.read_csv(path)
    sol["time"] = pd.to_datetime(sol["time"])
    sol = sol.set_index("time")["shortwave_radiation (W/m²)"]
    # Reindex to match the main dataset; fill gaps with 0
    sol_aligned = sol.reindex(main_index).fillna(0.0)
    return sol_aligned.values


# ===========================================================================
# Step 3 – simplified idle tank step (UA losses + inter-node conduction only)
# ===========================================================================
def idle_step(T: np.ndarray, T_amb: float, params: TankParams) -> np.ndarray:
    """Advance the tank by one 30-min timestep with no heat input.
    Uses the full tank_step function from src.tank_model with Q=0 for all
    sources, so UA_loss and UA_adj physics are applied consistently."""
    return tank_step(
        T,
        Q_st_kwh=0.0,
        Q_ashp_kwh=0.0,
        Q_imm_kwh=0.0,
        T_amb=T_amb,
        params=params,
        dt_s=DT_S,
    )


# ===========================================================================
# Step 4 – simplified free-forward simulator
#   At each timestep the precedence is:
#     1. Solar override  (if illuminance >= threshold AND not already at setpoint)
#     2. ASHP override   (if mid node <= trigger AND solar not just fired)
#     3. Idle physics    (UA losses + conduction)
# ===========================================================================
def simulate_simplified(
    T0: np.ndarray,
    T_amb: np.ndarray,
    solar_illum: np.ndarray,
    params: TankParams,
    solar_thresh: float,
    solar_setpoint: float,
    ashp_trigger: float,
    ashp_setpoint: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the simplified free-forward tank simulation over N steps.

    Parameters
    ----------
    T0            : initial temperatures, shape (4,).
    T_amb         : ambient temperature at each step, shape (N,).
    solar_illum   : solar illuminance at each step [W/m²], shape (N,).
    params        : TankParams with fitted UA_loss and UA_adj.
    solar_thresh  : illuminance threshold for solar activation [W/m²].
    solar_setpoint: temperature all nodes are set to when solar fires [°C].
    ashp_trigger  : mid-node temperature below which ASHP fires [°C].
    ashp_setpoint : temperature mid/mid-hi/top are set to when ASHP fires [°C].

    Returns
    -------
    T_hist      : (N+1, 4) – temperatures including T0.
    solar_flags : (N,) bool – True when solar override was applied.
    ashp_flags  : (N,) bool – True when ASHP override was applied.
    """
    N = len(T_amb)
    T_hist      = np.zeros((N + 1, 4))
    solar_flags = np.zeros(N, dtype=bool)
    ashp_flags  = np.zeros(N, dtype=bool)

    T_hist[0] = T0.copy()

    # solar_locked tracks whether we are inside a continuous above-threshold
    # illuminance period that has already been serviced by one solar fire.
    # The lock is released as soon as illuminance drops back below the
    # threshold, allowing the next distinct solar period to fire once.
    solar_locked = False

    for k in range(N):
        T_prev = T_hist[k]
        illum  = solar_illum[k]

        # -- Sub-step A: idle physics produces the baseline next state -------
        T_next = idle_step(T_prev, T_amb[k], params)

        # -- Sub-step B: solar override --------------------------------------
        # Release the lock when illuminance falls back below threshold so
        # the next distinct above-threshold period can fire once.
        if illum < solar_thresh:
            solar_locked = False

        # Fire once per continuous above-threshold period, and only when the
        # tank is not already at or above the set-point.
        if illum >= solar_thresh and not solar_locked and T_prev.max() < solar_setpoint:
            T_next[:] = solar_setpoint
            solar_flags[k] = True
            solar_locked = True  # suppress re-firing for remainder of this period

        # -- Sub-step C: ASHP override ---------------------------------------
        # Condition: mid node (index 1) at or below trigger temperature.
        # Solar takes priority; ASHP only checked when solar did not fire.
        elif T_next[1] <= ashp_trigger:
            T_next[1] = ashp_setpoint   # mid
            T_next[2] = ashp_setpoint   # mid-hi
            T_next[3] = ashp_setpoint   # top
            # Bottom node is unaffected (ASHP HX sits above bottom node)
            ashp_flags[k] = True

        T_hist[k + 1] = T_next

    return T_hist, solar_flags, ashp_flags


# ===========================================================================
# Step 5 – build shading spans helper (reused for solar and ASHP bands)
# ===========================================================================
def build_spans(flags: np.ndarray, times: pd.DatetimeIndex, dt: pd.Timedelta):
    """Convert a boolean flag array into (start, end) span tuples for
    axvspan shading.  Each True run is padded by one dt at the end so the
    final active interval is fully covered."""
    spans = []
    in_span = False
    for k, flag in enumerate(flags):
        if flag and not in_span:
            span_start = times[k]
            in_span = True
        elif not flag and in_span:
            spans.append((span_start, times[k]))
            in_span = False
    if in_span:
        spans.append((span_start, times[-1] + dt))
    return spans


# ===========================================================================
# Step 6 – main run function: loads data, simulates, computes RMSE, plots
# ===========================================================================
def run(
    start: str,
    end: str,
    csv: Path,
    yaml: Path,
    solar_csv: Path,
    global_json: Path,
    solar_thresh: float,
    solar_setpoint: float,
    ashp_trigger: float,
    ashp_setpoint: float,
):
    # -----------------------------------------------------------------------
    # 6a – load fitted parameters
    # -----------------------------------------------------------------------
    params = load_tank_params(global_json)
    print("\n=== Fitted UA parameters from {} ===".format(global_json.name))
    print("  UA_loss (kW/K) [bottom→top] : {}".format(
        np.round(params.UA_loss, 6).tolist()))
    print("  UA_adj  (kW/K) [b-m, m-mh, mh-t] : {}".format(
        np.round(params.UA_adj, 6).tolist()))

    print("\n=== Simplified activation thresholds ===")
    print("  Solar illuminance threshold : {:.1f} W/m²".format(solar_thresh))
    print("  Solar set-point (all nodes) : {:.1f} °C".format(solar_setpoint))
    print("  ASHP trigger (mid node ≤)   : {:.1f} °C".format(ashp_trigger))
    print("  ASHP set-point (mid/mh/top) : {:.1f} °C".format(ashp_setpoint))

    # -----------------------------------------------------------------------
    # 6b – load and slice the main 30-minute dataset
    # -----------------------------------------------------------------------
    df = load_and_clean(csv, yaml, sampling_minutes=30)
    mask = (df.index >= start) & (df.index <= end)
    win = df.loc[mask].copy()
    if len(win) < 2:
        sys.exit("ERROR: window contains only {} rows – check --start/--end.".format(
            len(win)))

    N = len(win)
    dt_index = win.index[1] - win.index[0]
    print("\n=== Window ===")
    print("  {} → {}  ({} intervals × 30 min)".format(
        win.index[0], win.index[-1], N))

    # -----------------------------------------------------------------------
    # 6c – align solar illuminance to the window
    # -----------------------------------------------------------------------
    solar_illum = load_solar_illuminance(solar_csv, win.index)
    n_zero_sol = int((solar_illum == 0).sum())
    n_above_thresh = int((solar_illum >= solar_thresh).sum())
    print("\n=== Solar illuminance ===")
    print("  Intervals with illuminance = 0          : {}".format(n_zero_sol))
    print("  Intervals ≥ threshold ({:.0f} W/m²) : {}".format(
        solar_thresh, n_above_thresh))

    # -----------------------------------------------------------------------
    # 6d – extract measured temperatures and ambient
    # -----------------------------------------------------------------------
    T_meas = win[NODE_COLS].values          # shape (N, 4)
    T_amb  = win["t_amb_c"].fillna(
        win["t_amb_c"].median()).values      # shape (N,)

    # -----------------------------------------------------------------------
    # 6e – run the simplified free-forward simulation
    # -----------------------------------------------------------------------
    T_hist, solar_flags, ashp_flags = simulate_simplified(
        T0=T_meas[0],
        T_amb=T_amb,
        solar_illum=solar_illum,
        params=params,
        solar_thresh=solar_thresh,
        solar_setpoint=solar_setpoint,
        ashp_trigger=ashp_trigger,
        ashp_setpoint=ashp_setpoint,
    )
    # T_hist[k+1] is the predicted state after processing interval k.
    # For plotting, T_hist[1:] aligns with win.index[0:] (each prediction is
    # placed at the timestamp of the interval that produced it).
    pred_T = T_hist[1:]   # shape (N, 4)

    n_solar_fired = int(solar_flags.sum())
    n_ashp_fired  = int(ashp_flags.sum())
    print("\n=== Simulation events ===")
    print("  Solar override fired : {} interval(s)".format(n_solar_fired))
    print("  ASHP override fired  : {} interval(s)".format(n_ashp_fired))

    # -----------------------------------------------------------------------
    # 6f – compute RMSE / bias for each node (compare pred[k] vs meas[k+1])
    # -----------------------------------------------------------------------
    # pred_T[:-1] = predictions for steps 0..N-2
    # T_meas[1:]  = measurements at steps 1..N-1
    # This is the standard 1-step-ahead comparison used throughout the repo.
    print("\n=== Free-simulation RMSE (excluding reset point) ===")
    for i, lbl in enumerate(NODE_LABELS):
        err = pred_T[:-1, i] - T_meas[1:, i]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        bias = float(np.mean(err))
        print("  {:<8}: RMSE={:.3f} °C   bias={:+.3f} °C".format(lbl, rmse, bias))

    # -----------------------------------------------------------------------
    # 6g – plot
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(
        5, 1, figsize=(14, 14), sharex=True,
        gridspec_kw={"height_ratios": [2, 2, 2, 2, 1.2]},
    )
    fig.suptitle(
        "Simplified solar model – free-forward validation\n"
        "Solar: illum ≥ {:.0f} W/m² → all nodes {:.1f} °C  |  "
        "ASHP: mid ≤ {:.1f} °C → mid/mh/top = {:.1f} °C\n"
        "{} → {}".format(
            solar_thresh, solar_setpoint,
            ashp_trigger, ashp_setpoint,
            win.index[0].strftime("%Y-%m-%d %H:%M"),
            win.index[-1].strftime("%Y-%m-%d %H:%M"),
        ),
        fontsize=10,
    )

    # Temperature panels – one per node (axes 0-3)
    for i in range(4):
        ax = axes[i]
        err_rmse = np.sqrt(np.mean((pred_T[:-1, i] - T_meas[1:, i]) ** 2))
        err_bias = np.mean(pred_T[:-1, i] - T_meas[1:, i])

        ax.plot(
            win.index, T_meas[:, i],
            "o-", color=NODE_COLOURS[i], linewidth=1.4, markersize=3.5,
            label="Measured", zorder=3,
        )
        ax.plot(
            win.index, pred_T[:, i],
            "--", color=NODE_COLOURS[i], linewidth=1.4, alpha=0.80,
            label="Predicted", zorder=3,
        )
        ax.set_ylabel("{} [°C]".format(NODE_LABELS[i]))
        ax.legend(
            title="RMSE={:.2f}°C  bias={:+.2f}°C".format(err_rmse, err_bias),
            loc="best", fontsize=7, title_fontsize=7,
        )
        ax.grid(True, alpha=0.3)

    # Solar illuminance panel (axis 4)
    ax_sol = axes[4]
    ax_sol.step(win.index, solar_illum, where="post",
                color="tab:orange", linewidth=1.2, label="Illuminance [W/m²]")
    ax_sol.axhline(solar_thresh, color="tab:orange", linestyle=":",
                   linewidth=1.0, label="Threshold ({:.0f} W/m²)".format(solar_thresh))
    ax_sol.set_ylabel("Shortwave\n[W/m²]")
    ax_sol.legend(loc="upper right", fontsize=7, ncol=2)
    ax_sol.grid(True, alpha=0.3)
    ax_sol.set_ylim(bottom=0)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
    axes[-1].set_xlabel("Time")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    _PLOT_OUT.mkdir(parents=True, exist_ok=True)
    tag = win.index[0].strftime("%Y%m%d_%H%M")
    out = _PLOT_OUT / "validate_simple_solar_{}.png".format(tag)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("\nPlot saved → {}".format(out))
    plt.close(fig)


# ===========================================================================
# Step 7 – CLI entry point
# ===========================================================================
def _parse_args():
    p = argparse.ArgumentParser(
        description="Validate simplified solar/ASHP activation model.")
    p.add_argument("--start",          default="2024-05-10 00:00",
                   help="Window start (YYYY-MM-DD HH:MM)")
    p.add_argument("--end",            default="2024-05-10 23:30",
                   help="Window end   (YYYY-MM-DD HH:MM)")
    p.add_argument("--csv",            default=str(_DEFAULT_CSV),
                   help="Path to 30-min plant CSV")
    p.add_argument("--yaml",           default=str(_DEFAULT_YAML),
                   help="Path to column-mapping YAML")
    p.add_argument("--solar-csv",      default=str(_SOLAR_ILL),
                   help="Path to Sol_Ill_23_25_30min.csv")
    p.add_argument("--global-json",    default=str(_GLOBAL_JSON),
                   help="Path to global_fit.json")
    p.add_argument("--solar-thresh",   type=float, default=180.0,
                   help="Solar activation threshold [W/m²]  (default 180)")
    p.add_argument("--solar-setpoint", type=float, default=61.5,
                   help="Solar set-point – all nodes [°C]   (default 61.5)")
    p.add_argument("--ashp-trigger",   type=float, default=45.0,
                   help="ASHP trigger – mid node ≤ this [°C] (default 45)")
    p.add_argument("--ashp-setpoint",  type=float, default=55.0,
                   help="ASHP set-point – mid/mh/top [°C]   (default 55)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        start=args.start,
        end=args.end,
        csv=Path(args.csv),
        yaml=Path(args.yaml),
        solar_csv=Path(args.solar_csv),
        global_json=Path(args.global_json),
        solar_thresh=args.solar_thresh,
        solar_setpoint=args.solar_setpoint,
        ashp_trigger=args.ashp_trigger,
        ashp_setpoint=args.ashp_setpoint,
    )
