#!/usr/bin/env python3
"""
scripts/validate_st_window.py - Free forward simulation over a solar charging window.

Validates the solar thermal regressions by running a 48-hour free forward
simulation with externally computed Q_sol and T_flow, then comparing predicted
vs measured node temperatures.

Solar Regressions (user-provided):
    Q_sol  = 0.0315 * ST_power - 0.00012    [kWh per interval, when active]
    T_flow = 0.026359 * Solar_Illum + 0.282368 * T_bottom + 30.03768  [°C]

Activation Gate:
    Solar_Illum > 180 W/m² AND T_bottom < 53.5 °C

Node Allocation:
    Q_sol[i] = (T_flow - T_i) / sum(T_flow - T_j) * Q_sol
    (proportional to thermal headroom; nodes at or above T_flow get zero)

Usage
-----
    python scripts/validate_st_window.py \
        --start "2024-05-08 00:00" --end "2024-05-09 23:30"
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
from src.solar_thermal import compute_st_energy
from src.tank_model import TankParams, NODE_CAP, simulate
from src.ashp_model import ASHPParams, predict_cop, sink_proxy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

# --- Paths ---
_GLOBAL_JSON = _ROOT / "Global_fitting" / "output" / "global_fit.json"
_ASHP_JSON = _ROOT / "ASHP_fitting" / "output" / "ashp_fit.json"
_DEFAULT_CSV = _ROOT / "data" / "FullDS_Findhorn_30min.csv"
_DEFAULT_YAML = _ROOT / "column_mapping.yaml"
_SOLAR_ILL_CSV = _ROOT / "data" / "Sol_Ill_23_25_30min.csv"
_PLOT_OUT = _ROOT / "ST_fitting" / "output" / "plots"

# --- Constants ---
NODE_LABELS = ["Bottom", "Mid", "Mid-Hi", "Top"]
NODE_COLS = ["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]
NODE_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# --- Solar regression coefficients (user-provided) ---
Q_SOL_SLOPE = 0.487798172      # kWh per kW of ST power
Q_SOL_INTERCEPT = -0.001699369 # kWh
T_FLOW_SOLAR_COEF = 0.026359   # °C per W/m²
T_FLOW_BOTTOM_COEF = 0.282368  # dimensionless
T_FLOW_INTERCEPT = 30.03768    # °C

# --- Activation thresholds ---
SOLAR_ILLUM_MIN = 180.0       # W/m²
T_BOTTOM_MAX = 53.5           # °C


def _load_tank_params(path: Path) -> TankParams:
    """Load TankParams from global_fit.json."""
    with open(path) as f:
        d = json.load(f)
    p = TankParams()
    p.UA_loss = np.array(d["UA_loss"], dtype=float)
    p.UA_adj = np.array(d["UA_adj"], dtype=float)
    p.f_st = np.array(d["f_st"], dtype=float)
    p.f_ashp = np.array(d["f_ashp"], dtype=float)
    p.f_imm = np.array(d["f_imm"], dtype=float)
    return p


def _load_ashp_params(path: Path) -> ASHPParams:
    """Load ASHPParams from ashp_fit.json."""
    with open(path) as f:
        d = json.load(f)
    a = d["ashp"]
    return ASHPParams(
        c=np.array(a["c"], dtype=float) if "c" in a else None,
        a=np.array(a["a"], dtype=float) if "a" in a else None,
        b=np.array(a["b"], dtype=float) if "b" in a else None,
    )


def solar_active(solar_illum: float, t_bottom: float) -> bool:
    """Check if solar thermal system should be modeled as active."""
    return solar_illum > SOLAR_ILLUM_MIN and t_bottom < T_BOTTOM_MAX


def compute_q_sol(st_power_kw: float) -> float:
    """Compute Q_sol [kWh] from ST power using user regression."""
    q = Q_SOL_SLOPE * st_power_kw + Q_SOL_INTERCEPT
    return max(0.0, q)


def compute_t_flow(solar_illum: float, t_bottom: float) -> float:
    """Compute T_flow [°C] from solar illuminance and bottom temperature."""
    return (T_FLOW_SOLAR_COEF * solar_illum +
            T_FLOW_BOTTOM_COEF * t_bottom +
            T_FLOW_INTERCEPT)


def compute_st_weights(t_nodes: np.ndarray, t_flow: float) -> np.ndarray:
    """Compute ST heat allocation weights based on thermal headroom.

    Each node receives heat proportional to max(T_flow - T_node, 0).
    Weights are normalized by the sum; if all nodes >= T_flow, all weights = 0.
    """
    dt = np.maximum(t_flow - np.asarray(t_nodes, dtype=float), 0.0)
    total = dt.sum()
    if total <= 0:
        return np.zeros(4)
    return dt / total


def load_solar_illuminance(path: Path) -> pd.Series:
    """Load solar illuminance CSV and return as Series with DatetimeIndex."""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    return df["shortwave_radiation (W/m²)"]


def run(
    start: str,
    end: str,
    csv: Path,
    yaml: Path,
    solar_csv: Path,
    global_json: Path,
    ashp_json: Path,
    sampling_minutes: int = 30,
):
    """Main verification workflow."""
    # Load parameters
    tank_params = _load_tank_params(global_json)
    ashp_params = _load_ashp_params(ashp_json)

    print("\nFitted parameters from {}:".format(global_json.name))
    print("  UA_loss : {} kW/K".format(np.round(tank_params.UA_loss, 5).tolist()))
    print("  UA_adj  : {} kW/K".format(np.round(tank_params.UA_adj, 5).tolist()))
    print("  f_ashp  : {}".format(tank_params.f_ashp.tolist()))

    dt_h = sampling_minutes / 60.0
    dt_s = sampling_minutes * 60.0

    # Load main dataset
    df = load_and_clean(csv, yaml, sampling_minutes=sampling_minutes)
    df["st_kwh"] = compute_st_energy(df, dt_minutes=sampling_minutes)

    # Load solar illuminance and align
    solar_illum = load_solar_illuminance(solar_csv)
    print("\nSolar illuminance loaded: {} rows".format(len(solar_illum)))
    print("  Range: {} to {}".format(solar_illum.index.min(), solar_illum.index.max()))

    # Merge solar illuminance into main dataframe
    df["solar_illum"] = solar_illum.reindex(df.index)

    # Slice to verification window
    mask = (df.index >= start) & (df.index <= end)
    win = df.loc[mask].copy()
    if len(win) < 2:
        sys.exit("Window has only {} rows.".format(len(win)))

    # Check for missing solar illuminance
    n_missing_solar = win["solar_illum"].isna().sum()
    if n_missing_solar > 0:
        print("WARNING: {} timestamps missing solar illuminance data".format(
            n_missing_solar))
        win["solar_illum"] = win["solar_illum"].fillna(0.0)

    N = len(win)
    print("\nWindow: {} -> {} ({} intervals, dt={} h)".format(
        win.index[0], win.index[-1], N, dt_h))

    # --- Build inputs ---
    T_meas = win[NODE_COLS].values
    Q_st_meas = win["st_kwh"].fillna(0).values
    Q_imm = win["imm_tot_inst_kwh"].fillna(0).values
    T_amb = win["t_amb_c"].fillna(win["t_amb_c"].median()).values
    P_meas = win["ashp_inst_kwh"].fillna(0).values
    ST_power = win["st_power_kw"].fillna(0).values
    Solar_Illum = win["solar_illum"].values

    # --- Compute ASHP heat (same as validate_ashp_window) ---
    mid_diff = pd.Series(win[NODE_COLS[1]].values, index=win.index).diff().fillna(0.0).values
    top_diff = pd.Series(win[NODE_COLS[3]].values, index=win.index).diff().fillna(0.0).values
    # For 30-min data, threshold is ~1.0°C; adjust if needed
    threshold = 1.0 if sampling_minutes >= 30 else 0.25
    mid_rising = mid_diff > threshold
    top_rising = top_diff > threshold
    node_rising = mid_rising | top_rising

    ashp_on = P_meas > 0.07 if sampling_minutes >= 30 else P_meas > 0.016
    st_off = Q_st_meas <= 0.001
    imm_off = Q_imm <= 0.01
    ashp_dhw_gate = ashp_on & st_off & imm_off & node_rising
    ashp_visible = ashp_on & node_rising

    # Predict Q_ashp = P_meas * COP
    T_sink = sink_proxy(win[NODE_COLS[1]].values, win[NODE_COLS[3]].values)
    T_out = win["t_out_c"].fillna(win["t_out_c"].median()).values
    cop = predict_cop(T_out, T_sink, ashp_params)
    Q_ashp = np.where(ashp_dhw_gate, P_meas * cop, 0.0)

    # --- Prepare external solar drivers ---
    # For verification: compute Q_sol whenever measured ST power > 0
    # (the system was actually operating regardless of temperature gate)
    # The temperature gate affects T_flow-based allocation only
    Q_sol = np.zeros(N)
    T_flow = np.zeros(N)
    f_st_ext = np.zeros((N, 4))
    solar_active_mask = np.zeros(N, dtype=bool)

    # For first step, use measured T_bottom; thereafter use predicted
    # But for external driver prep, we use measured T at each step
    # (since we're validating regression against measured trajectory)
    for k in range(N):
        t_bottom_k = T_meas[k, 0]
        solar_illum_k = Solar_Illum[k]
        st_power_k = ST_power[k]

        # Compute Q_sol from measured ST power when power > 0
        # This captures actual solar operation regardless of temperature gate
        if st_power_k > 0:
            solar_active_mask[k] = True
            Q_sol[k] = compute_q_sol(st_power_k)
            # Compute T_flow for allocation (requires illuminance for regression)
            if solar_illum_k > 0:
                T_flow[k] = compute_t_flow(solar_illum_k, t_bottom_k)
            else:
                # Fallback: use a simple estimate based on ST operation
                T_flow[k] = t_bottom_k + 20.0  # typical collector rise
            f_st_ext[k] = compute_st_weights(T_meas[k], T_flow[k])
        elif solar_active(solar_illum_k, t_bottom_k):
            # Gate met but no ST power - system might be warming up
            solar_active_mask[k] = True
            Q_sol[k] = 0.0
            T_flow[k] = compute_t_flow(solar_illum_k, t_bottom_k)
            f_st_ext[k] = np.zeros(4)
        else:
            Q_sol[k] = 0.0
            T_flow[k] = 0.0
            f_st_ext[k] = np.zeros(4)

    # Print solar statistics
    n_solar = int(solar_active_mask.sum())
    total_Q_sol = float(Q_sol.sum())
    total_Q_ashp = float(Q_ashp.sum())
    total_Q_imm = float(Q_imm.sum())
    print("\n  Solar-active intervals : {} / {}".format(n_solar, N))
    print("  ASHP-on intervals      : {} / {}".format(int((Q_ashp > 0).sum()), N))
    print("  Q_sol total            : {:.2f} kWh".format(total_Q_sol))
    print("  Q_ashp total           : {:.2f} kWh".format(total_Q_ashp))
    print("  Q_imm total            : {:.2f} kWh".format(total_Q_imm))

    # --- Free forward simulation ---
    T_hist = simulate(
        T_meas[0], Q_sol, Q_ashp, Q_imm, T_amb, tank_params,
        dt_s=dt_s, f_st_ext=f_st_ext,
    )
    T_pred = T_hist[1:]  # (N, 4) - prediction one step ahead of each input

    # RMSE against measured[1:] (excludes the reset point)
    T_meas_cmp = T_meas[1:]
    T_pred_cmp = T_pred[:N - 1]
    print("\nFree-simulation RMSE (excluding reset point):")
    for i, lbl in enumerate(NODE_LABELS):
        err = T_pred_cmp[:, i] - T_meas_cmp[:, i]
        print("  {:<8}: RMSE={:.3f} C  bias={:+.3f} C".format(
            lbl, float(np.sqrt(np.mean(err ** 2))), float(np.mean(err))))

    # --- Plot ---
    fig, axes = plt.subplots(6, 1, figsize=(14, 15), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 2, 2, 1.2, 1.2]})
    fig.suptitle(
        "Free forward simulation (Solar + ASHP verification)\n{} to {}".format(
            win.index[0].strftime("%Y-%m-%d %H:%M"),
            win.index[-1].strftime("%Y-%m-%d %H:%M")),
        fontsize=12,
    )

    # Build shading spans for ASHP
    dt_index = win.index[1] - win.index[0]
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

    # Build shading spans for Solar
    solar_spans = []
    in_span = False
    for k in range(N):
        if solar_active_mask[k] and not in_span:
            span_start = win.index[k]
            in_span = True
        elif not solar_active_mask[k] and in_span:
            solar_spans.append((span_start, win.index[k]))
            in_span = False
    if in_span:
        solar_spans.append((span_start, win.index[-1] + dt_index))

    # Temperature panels (0-3)
    pred_times = win.index
    pred_T = T_hist[1:N + 1]

    for i in range(4):
        ax = axes[i]
        ax.plot(win.index, T_meas[:, i],
                "o-", color=NODE_COLOURS[i], linewidth=1.4, markersize=3.5,
                label="Measured", zorder=3)
        ax.plot(pred_times, pred_T[:, i],
                "--", color=NODE_COLOURS[i], linewidth=1.4, alpha=0.8,
                label="Predicted", zorder=3)
        # ASHP shading (green)
        for (s, e) in ashp_spans:
            ax.axvspan(s, e, alpha=0.12, color="tab:green", zorder=1)
        # Solar shading (orange)
        for (s, e) in solar_spans:
            ax.axvspan(s, e, alpha=0.12, color="tab:orange", zorder=1)

        err_rmse = np.sqrt(np.mean((pred_T[:-1, i] - T_meas[1:, i]) ** 2))
        err_bias = np.mean(pred_T[:-1, i] - T_meas[1:, i])
        ax.set_ylabel("{} [°C]".format(NODE_LABELS[i]))
        ax.legend(
            title="RMSE={:.2f}°C  bias={:+.2f}°C".format(err_rmse, err_bias),
            loc="best", fontsize=7, title_fontsize=7,
        )
        ax.grid(True, alpha=0.3)

    # ASHP heat input panel (4)
    ax5 = axes[4]
    dt_min = dt_h * 60
    ax5.step(win.index, Q_ashp, where="post", color="tab:green",
             linewidth=1.2, label="Q_ashp")
    ax5.step(win.index, Q_imm, where="post", color="tab:red",
             linewidth=1.2, label="Q_imm")
    # ASHP shading
    for (s, e) in ashp_spans:
        ax5.axvspan(s, e, alpha=0.12, color="tab:green", zorder=1)
    ax5.set_ylabel("Heat input\n[kWh / {:.0f} min]".format(dt_min))
    ax5.legend(loc="upper right", fontsize=7, ncol=2)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(bottom=0)

    # ST power panel (5)
    ax6 = axes[5]
    ax6.step(win.index, ST_power, where="post", color="tab:orange",
             linewidth=1.2, label="ST Power [kW]")
    ax6.step(win.index, Q_sol, where="post", color="tab:blue",
             linewidth=1.2, linestyle="--", label="Q_sol [kWh]")
    # Solar shading
    for (s, e) in solar_spans:
        ax6.axvspan(s, e, alpha=0.12, color="tab:orange", zorder=1)
    ax6.set_ylabel("ST Power / Q_sol")
    ax6.legend(loc="upper right", fontsize=7, ncol=2)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(bottom=0)

    # Add legend patches for shading
    ashp_patch = Patch(color="tab:green", alpha=0.25, label="ASHP on")
    solar_patch = Patch(color="tab:orange", alpha=0.25, label="Solar active")
    axes[0].legend(
        handles=axes[0].get_legend_handles_labels()[0] + [ashp_patch, solar_patch],
        labels=axes[0].get_legend_handles_labels()[1] + ["ASHP on", "Solar active"],
        title="RMSE={:.2f}°C  bias={:+.2f}°C".format(
            float(np.sqrt(np.mean((pred_T[:-1, 0] - T_meas[1:, 0]) ** 2))),
            float(np.mean(pred_T[:-1, 0] - T_meas[1:, 0]))),
        loc="best", fontsize=7, title_fontsize=7,
    )

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
    axes[-1].set_xlabel("Time")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    _PLOT_OUT.mkdir(parents=True, exist_ok=True)
    tag = win.index[0].strftime("%Y%m%d_%H%M")
    out = _PLOT_OUT / "st_window_{}.png".format(tag)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("\nPlot saved to: {}".format(out))
    plt.close(fig)

    # Save diagnostics CSV
    diag = pd.DataFrame({
        "time": win.index,
        "solar_illum": Solar_Illum,
        "st_power_kw": ST_power,
        "solar_active": solar_active_mask,
        "Q_sol": Q_sol,
        "T_flow": T_flow,
        "w_bottom": f_st_ext[:, 0],
        "w_mid": f_st_ext[:, 1],
        "w_mid_hi": f_st_ext[:, 2],
        "w_top": f_st_ext[:, 3],
        "T_bottom_meas": T_meas[:, 0],
        "T_mid_meas": T_meas[:, 1],
        "T_mid_hi_meas": T_meas[:, 2],
        "T_top_meas": T_meas[:, 3],
        "T_bottom_pred": pred_T[:, 0],
        "T_mid_pred": pred_T[:, 1],
        "T_mid_hi_pred": pred_T[:, 2],
        "T_top_pred": pred_T[:, 3],
    })
    diag_path = _PLOT_OUT.parent / "diagnostics" / "st_window_{}_diag.csv".format(tag)
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    diag.to_csv(diag_path, index=False)
    print("Diagnostics saved to: {}".format(diag_path))


def _parse():
    p = argparse.ArgumentParser(
        description="Validate solar thermal regressions over a specified window.")
    p.add_argument("--start", default="2024-05-08 00:00",
                   help="Start timestamp (default: 2024-05-08 00:00)")
    p.add_argument("--end", default="2024-05-09 23:30",
                   help="End timestamp (default: 2024-05-09 23:30)")
    p.add_argument("--csv", default=str(_DEFAULT_CSV),
                   help="Path to main dataset CSV")
    p.add_argument("--yaml", default=str(_DEFAULT_YAML),
                   help="Path to column mapping YAML")
    p.add_argument("--solar-csv", default=str(_SOLAR_ILL_CSV),
                   help="Path to solar illuminance CSV")
    p.add_argument("--json", default=str(_GLOBAL_JSON),
                   help="Path to global_fit.json for tank params")
    p.add_argument("--ashp-json", default=str(_ASHP_JSON),
                   help="Path to ashp_fit.json for ASHP params")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    run(
        args.start,
        args.end,
        Path(args.csv),
        Path(args.yaml),
        Path(args.solar_csv),
        Path(args.json),
        Path(args.ashp_json),
        sampling_minutes=30,
    )
