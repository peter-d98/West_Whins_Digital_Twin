#!/usr/bin/env python3
"""
scripts/validate_window.py
--------------------------
Free-forward simulation of the 4-node tank over one or more user-specified
time windows.  Combines:

  * idle physics from ``Global_fitting/output/global_fit.json``
    (UA_loss, UA_adj),
  * a stochastic DHW demand drawn from the monthly profile fitted by
    ``DHW_fitting/`` (``dhw_profile.csv`` + per-month CV from
    ``dhw_daily_stats.csv``),
  * simplified solar-thermal and ASHP control rules.

For each window, the script writes a measured-vs-predicted PNG to
``scripts/plots/`` and prints RMSE / bias per node plus the number of
solar / ASHP fires.

Activation rules
----------------
  Solar rule  : when shortwave illuminance ≥ ``--solar-thresh`` W/m²
                (once per continuous above-threshold period, and only if
                the tank is not already above the setpoint), all four
                nodes are set to ``--solar-setpoint`` °C at the END of
                the timestep.

  ASHP rule   : when the predicted mid-node temperature is at or below
                ``--ashp-trigger`` °C, the upper three nodes are forced
                to a small upward stratification:
                    mid    = ashp_setpoint
                    mid_hi = ashp_setpoint + 0.5
                    top    = ashp_setpoint + 1.0
                The bottom node is unaffected by the ASHP override.

  DHW rule    : at every timestep, look up ``mean_V_l[month, slot]``
                from the fitted profile, scale by a *daily* lognormal
                factor s_d (sampled once per simulated day, using the
                per-month CV from ``dhw_daily_stats.csv``), and apply
                via ``src.tank_model.dhw_step`` using the propagation
                mode chosen by ``--dhw-mode``.

  Idle physics: between override events the tank evolves according to
                the fitted UA_loss / UA_adj.

Order per timestep:
    1. idle_step (UA losses + conduction)
    2. dhw_step  (draws happen continuously — applied BEFORE the
       heat-source checks so a draw that pulls mid below the ASHP
       trigger correctly triggers a charge in the same step)
    3. solar override (priority): if illum ≥ threshold, all 4 nodes
       set to solar_setpoint (overriding any preceding draw)
    4. else ASHP override: if mid ≤ trigger, upper-3 nodes set to the
       stratified ASHP setpoints (bottom unaffected)

Simulation modes
----------------
  Deterministic central run (always produced, drives the dashed
  "Predicted" line and the printed RMSE/bias): every day uses
  s_today = 1.0, so the trace is a single physically-coherent
  trajectory — sharp jumps to setpoint, smooth decays in between.

  Stochastic envelope (optional, ``--n-realisations > 1``): N runs with
  daily lognormal scale draws; the P05–P95 band is shaded behind the
  predicted line and the V_draw bar plot shows the across-realisation
  median draw per slot.

Usage
-----
    # Default: runs the two built-in example windows.
    python scripts/validate_window.py

    # Custom windows (one or more --window START END pairs):
    python scripts/validate_window.py \\
        --window "2024-01-15 00:00" "2024-01-16 23:30" \\
        --window "2024-07-15 00:00" "2024-07-16 23:30" \\
        --dhw-mode bottom_only --n-realisations 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_loader import load_and_clean
from src.tank_model import TankParams, tank_step, dhw_step

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_GLOBAL_JSON  = _ROOT / "Global_fitting" / "output" / "global_fit.json"
_DEFAULT_CSV  = _ROOT / "data" / "FullDS_Findhorn_30min.csv"
_DEFAULT_YAML = _ROOT / "column_mapping.yaml"
_SOLAR_ILL    = _ROOT / "data" / "Sol_Ill_23_25_30min.csv"
_DHW_PROFILE  = _ROOT / "DHW_fitting" / "output" / "dhw_profile.csv"
_DHW_DAILY    = _ROOT / "DHW_fitting" / "output" / "dhw_daily_stats.csv"
_PLOT_OUT     = _ROOT / "scripts" / "output" / "plots"

# ---------------------------------------------------------------------------
# Display settings
# ---------------------------------------------------------------------------
NODE_LABELS  = ["Bottom", "Mid", "Mid-Hi", "Top"]
NODE_COLS    = ["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]
NODE_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

DT_S = 1800.0  # 30-min in seconds


# ===========================================================================
# Loaders
# ===========================================================================
def load_tank_params(path: Path) -> TankParams:
    """Read UA_loss and UA_adj from global_fit.json."""
    with open(path) as fh:
        d = json.load(fh)
    p = TankParams()
    p.UA_loss = np.array(d["UA_loss"], dtype=float)
    p.UA_adj  = np.array(d["UA_adj"],  dtype=float)
    return p


def load_solar_illuminance(path: Path, main_index: pd.DatetimeIndex) -> np.ndarray:
    sol = pd.read_csv(path)
    sol["time"] = pd.to_datetime(sol["time"])
    sol = sol.set_index("time")["shortwave_radiation (W/m²)"]
    return sol.reindex(main_index).fillna(0.0).values


def load_dhw_profile(
    path: Path,
    zero_hours_start: float = 0.0,
    zero_hours_end: float = 0.0,
    min_volume_l: float = 0.0,
) -> dict[int, np.ndarray]:
    """Return {month: array(48) of mean_V_l}.

    If ``zero_hours_end > zero_hours_start``, slots whose start-time falls
    in the half-open interval ``[zero_hours_start, zero_hours_end)`` are
    forced to zero across all months.  Use 0,0 (default) for no override.

    If ``min_volume_l > 0``, any per-slot mean strictly below this volume
    is forced to zero (small-event suppression to remove low-confidence
    noise from the profile).
    """
    df = pd.read_csv(path)
    out = {}
    for month in range(1, 13):
        sub = df[df["month"] == month].sort_values("slot")
        arr = np.zeros(48)
        arr[sub["slot"].values] = sub["mean_V_l"].values
        out[month] = arr

    if zero_hours_end > zero_hours_start:
        slot_start = int(np.floor(zero_hours_start * 2))
        slot_end = int(np.ceil(zero_hours_end * 2))   # exclusive
        n_zeroed_per_month = slot_end - slot_start
        total_removed = 0.0
        for month in range(1, 13):
            total_removed += float(out[month][slot_start:slot_end].sum())
            out[month][slot_start:slot_end] = 0.0
        annual_l_removed = total_removed * 365.25 / 12.0
        print(f"  Zeroed slots [{slot_start}:{slot_end}] "
              f"({zero_hours_start:.1f}–{zero_hours_end:.1f} h) in profile — "
              f"removed ≈ {annual_l_removed:.0f} L/yr "
              f"({n_zeroed_per_month} slots/month × 12)")

    if min_volume_l > 0.0:
        n_below = 0
        total_removed = 0.0
        for month in range(1, 13):
            mask = (out[month] > 0.0) & (out[month] < min_volume_l)
            n_below += int(mask.sum())
            total_removed += float(out[month][mask].sum())
            out[month][mask] = 0.0
        annual_l_removed = total_removed * 365.25 / 12.0
        print(f"  Min-volume floor {min_volume_l:.1f} L: zeroed {n_below} "
              f"(month, slot) cells, removed ≈ {annual_l_removed:.0f} L/yr")
    return out


def load_dhw_cv(path: Path) -> dict[int, float]:
    """Return {month: coefficient of variation of daily total volume}.

    Months absent from the file (or with NaN CV) default to 0.4 — the
    median CV across all months in the West Whins dataset.
    """
    df = pd.read_csv(path)
    cv = {int(r["month"]): float(r["cv"]) for _, r in df.iterrows()
          if np.isfinite(r["cv"])}
    for m in range(1, 13):
        cv.setdefault(m, 0.4)
    return cv


# ===========================================================================
# Stochastic daily scale factor (lognormal, mean=1, CV=cv_month)
# ===========================================================================
def sample_daily_scale(cv: float, rng: np.random.Generator) -> float:
    """Draw a lognormal scale factor with E[s]=1 and CV[s]=cv.

    For X = LogNormal(mu, sigma):
        E[X]   = exp(mu + sigma^2/2)
        Var[X] = (exp(sigma^2) - 1) * E[X]^2
        => CV  = sqrt(exp(sigma^2) - 1)
        => sigma = sqrt(ln(1 + cv^2))
        => mu = -sigma^2/2  (to make E[X] = 1)
    """
    if cv <= 0.0:
        return 1.0
    sigma = float(np.sqrt(np.log(1.0 + cv * cv)))
    mu = -0.5 * sigma * sigma
    return float(rng.lognormal(mean=mu, sigma=sigma))


# ===========================================================================
# Idle step (UA only, no Q inputs — overrides handle heat injection)
# ===========================================================================
def idle_step(T: np.ndarray, T_amb: float, params: TankParams) -> np.ndarray:
    return tank_step(
        T,
        Q_st_kwh=0.0, Q_ashp_kwh=0.0, Q_imm_kwh=0.0,
        T_amb=T_amb, params=params, dt_s=DT_S,
    )


# ===========================================================================
# Free-forward simulation with DHW demand
# ===========================================================================
def simulate(
    T0: np.ndarray,
    times: pd.DatetimeIndex,
    T_amb: np.ndarray,
    solar_illum: np.ndarray,
    profile_by_month: dict[int, np.ndarray],
    cv_by_month: dict[int, float],
    params: TankParams,
    *,
    solar_thresh: float,
    solar_setpoint: float,
    ashp_trigger: float,
    ashp_setpoint: float,
    t_mains: float,
    rng: np.random.Generator,
    dhw_mode: str = "cascade",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the simulation; return T_hist, solar_flags, ashp_flags,
    V_draw_applied, daily_scale_used."""
    N = len(times)
    T_hist = np.zeros((N + 1, 4))
    T_hist[0] = T0.copy()
    solar_flags = np.zeros(N, dtype=bool)
    ashp_flags = np.zeros(N, dtype=bool)
    V_draws = np.zeros(N)
    daily_scale = np.ones(N)

    solar_locked = False
    current_date = None
    s_today = 1.0

    for k in range(N):
        ts = times[k]
        date = ts.normalize()
        if date != current_date:
            current_date = date
            cv = cv_by_month.get(int(ts.month), 0.4)
            s_today = sample_daily_scale(cv, rng)
        daily_scale[k] = s_today

        T_prev = T_hist[k]
        illum = solar_illum[k]

        # 1. Idle baseline
        T_next = idle_step(T_prev, T_amb[k], params)

        # 2. DHW draw — applied BEFORE the heat-source checks so a draw
        #    that pulls mid below the ASHP trigger correctly fires a
        #    charge in the same step.  Physically: draws happen
        #    continuously; the ASHP/ST controllers react to the resulting
        #    tank state.
        slot = ts.hour * 2 + ts.minute // 30
        mean_V = profile_by_month[int(ts.month)][slot]
        V_draw = s_today * float(mean_V)
        if V_draw > 0.0:
            T_next = dhw_step(T_next, V_draw, T_mains_c=t_mains, mode=dhw_mode)
        V_draws[k] = V_draw

        # 3. Solar override (one fire per continuous above-threshold period)
        if illum < solar_thresh:
            solar_locked = False
        if (illum >= solar_thresh and not solar_locked
                and T_prev.max() < solar_setpoint):
            T_next[:] = solar_setpoint
            solar_flags[k] = True
            solar_locked = True
        # 4. ASHP override (only if solar didn't fire).  Apply with small
        #    upward stratification so the post-fire state preserves the
        #    physical ordering top > mid_hi > mid that's visible in the
        #    measured trace (HX is in the upper portion of the tank).
        elif T_next[1] <= ashp_trigger:
            T_next[1] = ashp_setpoint              # mid    = 55.0
            T_next[2] = ashp_setpoint + 0.5        # mid_hi = 55.5
            T_next[3] = ashp_setpoint + 1.0        # top    = 56.0
            ashp_flags[k] = True

        T_hist[k + 1] = T_next

    return T_hist, solar_flags, ashp_flags, V_draws, daily_scale


# ===========================================================================
# Span helper for shading
# ===========================================================================
def build_spans(flags: np.ndarray, times: pd.DatetimeIndex, dt: pd.Timedelta):
    spans = []
    in_span = False
    span_start = None
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
# Plot one window with optional stochastic envelope + deterministic central run
# ===========================================================================
def plot_window(
    win: pd.DataFrame,
    realisations: list[np.ndarray],          # list of T_hist arrays, each (N+1, 4)
    T_central: np.ndarray,                   # deterministic central run, (N+1, 4)
    solar_flags: np.ndarray,                 # from central run
    ashp_flags: np.ndarray,                  # from central run
    V_draws_med: np.ndarray,                 # median V_draw per slot across realisations
    solar_illum: np.ndarray,
    out_path: Path,
    *,
    solar_thresh: float, solar_setpoint: float,
    ashp_trigger: float, ashp_setpoint: float,
    n_realisations: int,
    dhw_mode: str,
):
    """Render the 6-panel measured-vs-predicted window plot.

    Panel order, top → bottom:
        Top temp → Mid-Hi temp → Mid temp → Bottom temp
        → Solar illuminance → V_draw

    The four temperature panels show measured (solid + markers) and the
    deterministic central predicted trace (dashed); when N>1 stochastic
    realisations are present, a P05–P95 envelope is shaded behind the
    predicted line.  Solar fires shade the panel orange, ASHP fires
    shade it purple.
    """
    T_meas = win[NODE_COLS].values
    times = win.index
    dt = times[1] - times[0]

    # Envelope only computed when there are multiple realisations.
    if len(realisations) > 1:
        stacked = np.stack(realisations, axis=0)
        pred_p05 = np.percentile(stacked, 5, axis=0)[1:]
        pred_p95 = np.percentile(stacked, 95, axis=0)[1:]
    else:
        pred_p05 = pred_p95 = None
    pred_central = T_central[1:]                      # (N, 4) for plotting

    fig, axes = plt.subplots(
        6, 1, figsize=(14, 16), sharex=True,
        gridspec_kw={"height_ratios": [2, 2, 2, 2, 1.0, 1.0]},
    )
    # Panel layout (top \u2192 bottom): Top, Mid-Hi, Mid, Bottom, Solar, V_draw.
    # The four temperature panels are reversed relative to NODE_LABELS so
    # the stratification visible at a glance matches the physical tank.
    temp_axes = {
        3: axes[0],   # Top
        2: axes[1],   # Mid-Hi
        1: axes[2],   # Mid
        0: axes[3],   # Bottom
    }
    ax_sol = axes[4]
    ax_dhw = axes[5]

    env_msg = (f"P05\u2013P95 envelope from {n_realisations} stochastic realisations"
               if n_realisations > 1 else
               "deterministic central run only (envelope disabled)")
    fig.suptitle(
        f"Free-forward tank simulation \u2014 {times[0]:%Y-%m-%d %H:%M} \u2192 "
        f"{times[-1]:%Y-%m-%d %H:%M}\n"
        f"Solar: illum \u2265 {solar_thresh:.0f} W/m\u00b2 \u2192 all nodes "
        f"{solar_setpoint:.1f}\u00b0C  |  "
        f"ASHP: mid \u2264 {ashp_trigger:.1f}\u00b0C \u2192 "
        f"mid={ashp_setpoint:.1f}/mh={ashp_setpoint+0.5:.1f}/"
        f"top={ashp_setpoint+1.0:.1f}\u00b0C  |  "
        f"DHW mode: {dhw_mode}  |  {env_msg}",
        fontsize=10,
    )

    solar_spans = build_spans(solar_flags, times, dt)
    ashp_spans  = build_spans(ashp_flags,  times, dt)

    for i in range(4):
        ax = temp_axes[i]
        # Shading for override events
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

        # RMSE / bias on the central run, one-step-ahead alignment
        err = pred_central[:-1, i] - T_meas[1:, i]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        bias = float(np.mean(err))
        ax.set_ylabel(f"{NODE_LABELS[i]} [\u00b0C]")
        ax.legend(title=f"RMSE={rmse:.2f}\u00b0C  bias={bias:+.2f}\u00b0C",
                  loc="best", fontsize=7, title_fontsize=7)
        ax.grid(True, alpha=0.3)

    # Solar illuminance panel
    ax_sol.step(times, solar_illum, where="post",
                color="tab:orange", lw=1.2, label="Illuminance [W/m\u00b2]")
    ax_sol.axhline(solar_thresh, color="tab:orange", ls=":", lw=1.0,
                   label=f"Threshold ({solar_thresh:.0f} W/m\u00b2)")
    ax_sol.set_ylabel("Shortwave\n[W/m\u00b2]")
    ax_sol.legend(loc="upper right", fontsize=7, ncol=2)
    ax_sol.grid(True, alpha=0.3)
    ax_sol.set_ylim(bottom=0)

    # DHW draw panel (bottom)
    bar_label = ("DHW draw (median across realisations) [L]" if n_realisations > 1
                 else "DHW draw (central, s=1) [L]")
    ax_dhw.bar(times, V_draws_med, width=dt.total_seconds() / 86400.0,
               color="tab:blue", alpha=0.7, align="edge",
               label=bar_label)
    ax_dhw.set_ylabel("V_draw\n[L per 30 min]")
    ax_dhw.legend(loc="upper right", fontsize=7)
    ax_dhw.grid(True, alpha=0.3)
    ax_dhw.set_ylim(bottom=0)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
    axes[-1].set_xlabel("Time")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved \u2192 {out_path}")


# ===========================================================================
# Run one window
# ===========================================================================
def run_window(
    start: str, end: str,
    df_full: pd.DataFrame,
    solar_full: np.ndarray,
    profile_by_month: dict[int, np.ndarray],
    cv_by_month: dict[int, float],
    params: TankParams,
    *,
    solar_thresh: float, solar_setpoint: float,
    ashp_trigger: float, ashp_setpoint: float,
    t_mains: float,
    n_realisations: int,
    seed: int,
    out_dir: Path,
    dhw_mode: str = "cascade",
) -> None:
    mask = (df_full.index >= start) & (df_full.index <= end)
    win = df_full.loc[mask].copy()
    sol_win = solar_full[np.asarray(mask)]
    if len(win) < 2:
        print(f"WARNING: window {start} → {end} has only {len(win)} rows; "
              "skipping.")
        return

    T_amb = win["t_amb_c"].fillna(win["t_amb_c"].median()).values
    T_meas = win[NODE_COLS].values

    # --- Deterministic central run (s_today = 1.0 every day).  Achieved by
    # passing a zero-CV map: sample_daily_scale returns 1.0 when cv<=0.
    # This single physically-coherent trajectory is what the dashed
    # "predicted" line shows — sharp jumps to setpoint, smooth decays
    # between — instead of an across-realisation median that would smear
    # out the setpoint plateaus when ASHP fire timings drift between runs.
    cv_zero = {m: 0.0 for m in range(1, 13)}
    rng_det = np.random.default_rng(seed)  # unused (cv=0 -> deterministic)
    T_central, solar_flags_ref, ashp_flags_ref, V_draw_central, _ = simulate(
        T0=T_meas[0],
        times=win.index, T_amb=T_amb, solar_illum=sol_win,
        profile_by_month=profile_by_month,
        cv_by_month=cv_zero,
        params=params,
        solar_thresh=solar_thresh, solar_setpoint=solar_setpoint,
        ashp_trigger=ashp_trigger, ashp_setpoint=ashp_setpoint,
        t_mains=t_mains, rng=rng_det,
        dhw_mode=dhw_mode,
    )

    # --- Stochastic realisations for the P05-P95 envelope only.
    #     Skipped entirely when n_realisations <= 1 (envelope parked).
    realisations = []
    V_draws_all = []
    if n_realisations > 1:
        for r in range(n_realisations):
            rng = np.random.default_rng(seed + r + 1)
            T_hist, _, _, V_draws, _ = simulate(
                T0=T_meas[0],
                times=win.index, T_amb=T_amb, solar_illum=sol_win,
                profile_by_month=profile_by_month,
                cv_by_month=cv_by_month,
                params=params,
                solar_thresh=solar_thresh, solar_setpoint=solar_setpoint,
                ashp_trigger=ashp_trigger, ashp_setpoint=ashp_setpoint,
                t_mains=t_mains, rng=rng,
                dhw_mode=dhw_mode,
            )
            realisations.append(T_hist)
            V_draws_all.append(V_draws)
        V_draws_med = np.median(np.stack(V_draws_all, axis=0), axis=0)
    else:
        # Use the central run as the source for the V_draw bar plot.
        V_draws_med = V_draw_central

    # Console summary — RMSE/bias from the deterministic central trace
    pred_central = T_central[1:]
    print(f"\n=== {win.index[0]} → {win.index[-1]}  "
          f"({len(win)} intervals, central run"
          f"{f' + {n_realisations} stochastic realisations' if n_realisations > 1 else ''}) ===")
    for i, lbl in enumerate(NODE_LABELS):
        err = pred_central[:-1, i] - T_meas[1:, i]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        bias = float(np.mean(err))
        print(f"  {lbl:<8}: RMSE={rmse:.3f} °C   bias={bias:+.3f} °C")
    print(f"  Solar fires: {int(solar_flags_ref.sum())} | "
          f"ASHP fires: {int(ashp_flags_ref.sum())} | "
          f"Total DHW (central): {V_draw_central.sum():.1f} L")

    # Filename: window_<startTag>_to_<endTag>[_<dhw_mode>].png
    start_tag = win.index[0].strftime("%Y%m%d_%H%M")
    end_tag   = win.index[-1].strftime("%Y%m%d_%H%M")
    suffix = "" if dhw_mode == "cascade" else f"_{dhw_mode}"
    out = out_dir / f"window_{start_tag}_to_{end_tag}{suffix}.png"
    plot_window(
        win=win, realisations=realisations, T_central=T_central,
        solar_flags=solar_flags_ref, ashp_flags=ashp_flags_ref,
        V_draws_med=V_draws_med, solar_illum=sol_win,
        out_path=out,
        solar_thresh=solar_thresh, solar_setpoint=solar_setpoint,
        ashp_trigger=ashp_trigger, ashp_setpoint=ashp_setpoint,
        n_realisations=n_realisations,
        dhw_mode=dhw_mode,
    )


# ===========================================================================
# CLI
# ===========================================================================
_DEFAULT_WINDOWS = [
    ("2024-01-15 00:00", "2024-01-16 23:30"),
    ("2024-07-15 00:00", "2024-07-16 23:30"),
]


def _parse_args():
    p = argparse.ArgumentParser(
        description="Free-forward DHW + simplified solar/ASHP validation "
                    "over one or more user-specified time windows.")
    p.add_argument("--window", nargs=2, action="append",
                   metavar=("START", "END"),
                   help="A window to simulate, given as two timestamps "
                        "(e.g. '2024-01-15 00:00' '2024-01-16 23:30'). "
                        "Pass --window multiple times to simulate several "
                        "windows.  If omitted, two example windows are run "
                        "(2024-01-15/16 and 2024-07-15/16).")
    p.add_argument("--csv",          default=str(_DEFAULT_CSV))
    p.add_argument("--yaml",         default=str(_DEFAULT_YAML))
    p.add_argument("--solar-csv",    default=str(_SOLAR_ILL))
    p.add_argument("--global-json",  default=str(_GLOBAL_JSON))
    p.add_argument("--dhw-profile",  default=str(_DHW_PROFILE))
    p.add_argument("--dhw-daily",    default=str(_DHW_DAILY))
    p.add_argument("--solar-thresh",   type=float, default=180.0)
    p.add_argument("--solar-setpoint", type=float, default=62.0)
    p.add_argument("--ashp-trigger",   type=float, default=45.0)
    p.add_argument("--ashp-setpoint",  type=float, default=55.0)
    p.add_argument("--t-mains",        type=float, default=10.0)
    p.add_argument("--n-realisations", type=int,   default=1,
                   help="Number of stochastic realisations for the P05–P95 "
                        "envelope.  Default 1 (envelope disabled, central "
                        "deterministic run only).")
    p.add_argument("--zero-hours-start", type=float, default=0.0,
                   help="Start of the hard-zeroed quiet window (hours, 0–24).")
    p.add_argument("--zero-hours-end",   type=float, default=5.0,
                   help="End of the hard-zeroed quiet window (hours, 0–24).  "
                        "Default 5.0 — zeros draws between 00:00 and 05:00.")
    p.add_argument("--min-volume",       type=float, default=5.0,
                   help="Per-slot mean volumes below this threshold (L) are "
                        "forced to zero — suppresses low-confidence small "
                        "events likely to be detector noise.  Default 5.0.")
    p.add_argument("--dhw-mode", choices=["cascade", "bottom_only"],
                   default="cascade",
                   help="DHW draw propagation. 'cascade' (default): piston "
                        "flow bottom\u2192top. 'bottom_only': mains mixes "
                        "only into the bottom node; upper nodes feel the "
                        "draw only via UA_adj conduction.")
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


def main():
    args = _parse_args()
    windows = args.window if args.window else _DEFAULT_WINDOWS

    print(f"Loading tank params from {args.global_json} ...")
    params = load_tank_params(Path(args.global_json))
    print(f"  UA_loss = {np.round(params.UA_loss, 6).tolist()}")
    print(f"  UA_adj  = {np.round(params.UA_adj,  6).tolist()}")

    print(f"Loading DHW profile from {args.dhw_profile} ...")
    profile_by_month = load_dhw_profile(
        Path(args.dhw_profile),
        zero_hours_start=args.zero_hours_start,
        zero_hours_end=args.zero_hours_end,
        min_volume_l=args.min_volume,
    )
    cv_by_month = load_dhw_cv(Path(args.dhw_daily))
    print(f"  Per-month daily-volume CV: "
          f"{ {m: round(cv_by_month[m], 2) for m in range(1, 13)} }")

    print(f"Loading data from {args.csv} ...")
    df = load_and_clean(args.csv, args.yaml, sampling_minutes=30)
    sol_full = load_solar_illuminance(Path(args.solar_csv), df.index)

    _PLOT_OUT.mkdir(parents=True, exist_ok=True)

    common = dict(
        df_full=df, solar_full=sol_full,
        profile_by_month=profile_by_month,
        cv_by_month=cv_by_month,
        params=params,
        solar_thresh=args.solar_thresh, solar_setpoint=args.solar_setpoint,
        ashp_trigger=args.ashp_trigger, ashp_setpoint=args.ashp_setpoint,
        t_mains=args.t_mains,
        n_realisations=args.n_realisations,
        seed=args.seed,
        out_dir=_PLOT_OUT,
        dhw_mode=args.dhw_mode,
    )
    for start, end in windows:
        run_window(start, end, **common)


if __name__ == "__main__":
    main()
