#!/usr/bin/env python3
"""
scripts/validate_ashp_window.py - Free forward simulation over an ASHP charging window.

Resets to measured temperatures at window start then simulates continuously
(no mid-window resets).  Overlays measured vs predicted temperatures and
marks ASHP-on intervals with shading.  A bottom panel shows modelled heat
inputs (Q_ashp, Q_st, Q_imm) per interval.

Usage
-----
    python scripts/validate_ashp_window.py \
        --start "2024-12-02 13:00" --end "2024-12-03 10:30"
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_loader import load_and_clean
from src.solar_thermal import compute_st_energy
from src.tank_model import TankParams, NODE_CAP, simulate
from src.ashp_model import ASHPParams, predict_cop, sink_proxy

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

_GLOBAL_JSON  = _ROOT / "Global_fitting" / "output" / "global_fit.json"
_ASHP_JSON    = _ROOT / "ASHP_fitting" / "output" / "ashp_fit.json"
_DEFAULT_CSV  = _ROOT / "data" / "FullDS_Findhorn_5min.csv"
_DEFAULT_YAML = _ROOT / "column_mapping_5min.yaml"
_PLOT_OUT     = _ROOT / "Global_fitting" / "output" / "plots"

NODE_LABELS  = ["Bottom", "Mid", "Mid-Hi", "Top"]
NODE_COLS    = ["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]
NODE_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def _load_tank_params(path):
    with open(path) as f: d = json.load(f)
    p = TankParams()
    p.UA_loss = np.array(d["UA_loss"], dtype=float)
    p.UA_adj  = np.array(d["UA_adj"],  dtype=float)
    p.f_st    = np.array(d["f_st"],    dtype=float)
    p.f_ashp  = np.array(d["f_ashp"],  dtype=float)
    p.f_imm   = np.array(d["f_imm"],   dtype=float)
    return p


def _load_ashp_params(path):
    with open(path) as f: d = json.load(f)
    a = d["ashp"]
    return ASHPParams(
        c=np.array(a["c"], dtype=float) if "c" in a else None,
        a=np.array(a["a"], dtype=float) if "a" in a else None,
        b=np.array(a["b"], dtype=float) if "b" in a else None,
    )


def run(start, end, csv, yaml, global_json, ashp_json, sampling_minutes=5, backcalc=False):
    tank_params = _load_tank_params(global_json)
    ashp_params = _load_ashp_params(ashp_json)

    print("\nFitted parameters from {}:".format(global_json.name))
    print("  UA_loss : {} kW/K".format(np.round(tank_params.UA_loss, 5).tolist()))
    print("  UA_adj  : {} kW/K".format(np.round(tank_params.UA_adj,  5).tolist()))
    print("  f_ashp  : {}".format(tank_params.f_ashp.tolist()))
    if backcalc:
        print("  ** Q_ashp mode: BACK-CALCULATED from nodes 1-3 energy balance **")
    else:
        print("  ** Q_ashp mode: PREDICTED (P_meas x COP map) — MPC-relevant **")

    dt_h = sampling_minutes / 60.0
    dt_s = sampling_minutes * 60.0

    df = load_and_clean(csv, yaml, sampling_minutes=sampling_minutes)
    df["st_kwh"] = compute_st_energy(df, dt_minutes=sampling_minutes)

    mask = (df.index >= start) & (df.index <= end)
    win = df.loc[mask].copy()
    if len(win) < 2:
        sys.exit("Window has only {} rows.".format(len(win)))

    N = len(win)
    print("\nWindow: {} -> {}  ({} intervals, dt={} h)".format(
        win.index[0], win.index[-1], N, dt_h))

    # --- Build inputs using the same logic as fit_global._prepare_inputs ---
    T_meas  = win[NODE_COLS].values
    Q_st    = win["st_kwh"].fillna(0).values
    Q_imm   = win["imm_tot_inst_kwh"].fillna(0).values
    T_amb   = win["t_amb_c"].fillna(win["t_amb_c"].median()).values

    P_meas  = win["ashp_inst_kwh"].fillna(0).values
    # mid_rising / top_rising thresholds are tuned for 5-min data.
    # 0.25 °C/5-min ≈ 3 °C/hour — indicates active DHW heating vs passive drift.
    # (The old 1.0 °C threshold was calibrated for 30-min data; at 5-min it only
    # caught the initial stratification-collapse burst and missed steady charging.)
    mid_diff = pd.Series(win["tank_mid_c"].values, index=win.index).diff().fillna(0.0).values
    top_diff = pd.Series(win["tank_top_c"].values, index=win.index).diff().fillna(0.0).values
    mid_rising = mid_diff > 0.25
    top_rising = top_diff > 0.25
    node_rising = mid_rising | top_rising
    # Shading: ASHP visibly on (power + at least one upper node rising).
    # Threshold matches pipeline cfg.ashp_off_kwh = 0.016 kWh.
    ashp_visible = (P_meas > 0.016) & node_rising

    # Gate: ASHP drawing power AND solar-thermal / immersion off
    # AND at least one upper node rising (separates DHW from space-heating).
    ashp_on = P_meas > 0.016
    st_off  = Q_st <= 0.001
    imm_off = Q_imm <= 0.01
    ashp_dhw_gate = ashp_on & st_off & imm_off & node_rising

    if backcalc:
        # Back-calculate Q_ashp from the measured energy balance on nodes 1–3.
        # Useful as a diagnostic to test the tank model with "known" heat.
        Q_ashp = np.zeros(N)
        for k in range(1, N):
            if not ashp_dhw_gate[k]:
                continue
            storage_kJ = sum(
                NODE_CAP[i] * (T_meas[k, i] - T_meas[k - 1, i])
                for i in range(1, 4)
            )
            loss_kJ = sum(
                tank_params.UA_loss[i] * (T_meas[k - 1, i] - T_amb[k]) * dt_s
                for i in range(1, 4)
            )
            Q_kJ = storage_kJ + loss_kJ
            Q_ashp[k] = max(Q_kJ / 3600.0, 0.0)
    else:
        # Predicted Q_ashp = P_meas × COP map — MPC-relevant forward prediction.
        T_sink = sink_proxy(win["tank_mid_c"].values, win["tank_top_c"].values)
        T_out  = win["t_out_c"].fillna(win["t_out_c"].median()).values
        cop    = predict_cop(T_out, T_sink, ashp_params)
        Q_ashp = np.where(ashp_dhw_gate, P_meas * cop, 0.0)

    n_ashp = int((Q_ashp > 0).sum())
    total_Q_ashp = float(Q_ashp.sum())
    total_Q_st   = float(Q_st.sum())
    total_Q_imm  = float(Q_imm.sum())
    print("  ASHP-on intervals : {} / {}".format(n_ashp, N))
    print("  Q_ashp total      : {:.2f} kWh".format(total_Q_ashp))
    print("  Q_st   total      : {:.2f} kWh".format(total_Q_st))
    print("  Q_imm  total      : {:.2f} kWh".format(total_Q_imm))

    # --- Free forward simulation from t=0 (no mid-window resets) ---
    T_hist = simulate(
        T_meas[0], Q_st, Q_ashp, Q_imm, T_amb, tank_params, dt_s=dt_s,
    )
    # T_hist shape (N+1, 4); T_hist[k] = state after k steps
    T_pred = T_hist[1:]   # (N, 4) — prediction one step ahead of each input

    # RMSE against measured[1:] (excludes the reset point)
    T_meas_cmp = T_meas[1:]
    T_pred_cmp = T_pred[:N-1]
    print("\nFree-simulation RMSE (excluding reset point):")
    for i, lbl in enumerate(NODE_LABELS):
        err = T_pred_cmp[:, i] - T_meas_cmp[:, i]
        print("  {:<8}: RMSE={:.3f} C  bias={:+.3f} C".format(
            lbl, float(np.sqrt(np.mean(err**2))), float(np.mean(err))))

    # --- Plot ---
    fig, axes = plt.subplots(5, 1, figsize=(14, 13), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 2, 2, 1.2]})
    fig.suptitle(
        "Free forward simulation (ASHP charging window){}\n{} to {}".format(
            " [Q_ashp = back-calc nodes 1-3]" if backcalc else " [Q_ashp = P_meas x COP]",
            win.index[0].strftime("%Y-%m-%d %H:%M"),
            win.index[-1].strftime("%Y-%m-%d %H:%M")),
        fontsize=12,
    )

    # Shading uses mid_rising (earliest visible sign of ASHP activity).
    # Span ends are extended by one dt so the final on-interval is fully covered.
    dt_index = win.index[1] - win.index[0]  # infer interval length from data
    ashp_spans = []
    in_span = False
    for k in range(N):
        if ashp_visible[k] and not in_span:
            span_start = win.index[k]
            in_span = True
        elif not ashp_visible[k] and in_span:
            ashp_spans.append((span_start, win.index[k]))
            in_span = False
    if in_span:
        ashp_spans.append((span_start, win.index[-1] + dt_index))

    # Temperature panels (indices 0-3)
    # T_hist[0] = reset = T_meas[0], then T_hist[1..N] are predictions
    # Align: win.index[k] corresponds to T_meas[k] and T_hist[k+1] (the prediction)
    # Show measured at all N points; show predicted starting from the
    # second point (T_hist[1..N]) — first point is shared as the reset.
    pred_times = win.index  # T_hist[1..N] predicted VALUES plotted at win.index[0..N-1]
    pred_T = T_hist[1:N+1]  # shape (N, 4)

    for i in range(4):
        ax = axes[i]
        ax.plot(win.index, T_meas[:, i],
                "o-", color=NODE_COLOURS[i], linewidth=1.4, markersize=3.5,
                label="Measured", zorder=3)
        ax.plot(pred_times, pred_T[:, i],
                "--", color=NODE_COLOURS[i], linewidth=1.4, alpha=0.8,
                label="Predicted", zorder=3)
        for (s, e) in ashp_spans:
            ax.axvspan(s, e, alpha=0.12, color="tab:green", zorder=1)
        # Compare free-sim prediction for time k+1 (pred_T[:-1] = T_hist[1:N-1])
        # against measurement at time k+1 (T_meas[1:]).  Consistent with the
        # terminal RMSE.  (pred_T[1:] would be off by one step.)
        err_rmse = np.sqrt(np.mean((pred_T[:-1, i] - T_meas[1:, i])**2))
        err_bias = np.mean(pred_T[:-1, i] - T_meas[1:, i])
        ax.set_ylabel("{} [C]".format(NODE_LABELS[i]))
        ax.legend(
            title="RMSE={:.2f}C  bias={:+.2f}C".format(err_rmse, err_bias),
            loc="best", fontsize=7, title_fontsize=7,
        )
        ax.grid(True, alpha=0.3)

    # Heat input panel (index 4)
    ax5 = axes[4]
    dt_min = dt_h * 60
    ax5.step(win.index, Q_ashp, where="post", color="tab:green",
             linewidth=1.2, label="Q_ashp")
    ax5.step(win.index, Q_st,   where="post", color="tab:orange",
             linewidth=1.2, label="Q_st")
    ax5.step(win.index, Q_imm,  where="post", color="tab:red",
             linewidth=1.2, label="Q_imm")
    ax5.set_ylabel("Heat input\n[kWh / {:.0f} min]".format(dt_min))
    ax5.legend(loc="upper right", fontsize=7, ncol=3)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(bottom=0)

    shading_patch = Patch(color="tab:green", alpha=0.25, label="ASHP DHW on")
    axes[0].legend(
        handles=axes[0].get_legend_handles_labels()[0] + [shading_patch],
        labels=axes[0].get_legend_handles_labels()[1] + ["ASHP DHW on"],
        title="RMSE={:.2f}C  bias={:+.2f}C".format(
            float(np.sqrt(np.mean((pred_T[:-1, 0] - T_meas[1:, 0])**2))),
            float(np.mean(pred_T[:-1, 0] - T_meas[1:, 0]))),
        loc="best", fontsize=7, title_fontsize=7,
    )

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
    axes[-1].set_xlabel("Time")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    _PLOT_OUT.mkdir(parents=True, exist_ok=True)
    tag = win.index[0].strftime("%Y%m%d_%H%M")
    suffix = "_backcalc" if backcalc else ""
    out = _PLOT_OUT / "ashp_window_{}{}.png".format(tag, suffix)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("\nPlot saved to: {}".format(out))
    plt.close(fig)


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2024-12-02 13:00")
    p.add_argument("--end",   default="2024-12-03 10:30")
    p.add_argument("--csv",   default=str(_DEFAULT_CSV))
    p.add_argument("--yaml",  default=str(_DEFAULT_YAML))
    p.add_argument("--json",  default=str(_GLOBAL_JSON))
    p.add_argument("--ashp-json", default=str(_ASHP_JSON))
    p.add_argument("--backcalc", action="store_true",
                   help="Use back-calculated Q_ashp from measured temperatures "
                        "instead of gated P_meas x COP.")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse()
    run(args.start, args.end, Path(args.csv), Path(args.yaml),
        Path(args.json), Path(args.ashp_json), backcalc=args.backcalc)
