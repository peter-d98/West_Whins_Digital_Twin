# ST_fitting – Solar-Thermal Heat Distribution Identification

## Overview

This tool identifies clean ST-only intervals in 5-minute plant data and
computes the empirical heat-distribution vector `f_st` — the fraction of
solar-thermal energy that appears in each of the 4 tank nodes within a
single time step.

The ST coil is physically located in the bottom node of the 550 L DHW tank,
but convective circulation and inter-node conduction cause some heat to
appear in higher nodes.  Since the grey-box model does not simulate genuine
water circulation, `f_st` captures this effective heat spreading empirically.

The method is consistent with how `f_ashp` is derived in `ASHP_fitting`:
measure the per-node stored energy change over clean single-source windows,
and take the normalised median across all valid windows.

## Physics

### ST-only interval detection

An interval qualifies as "ST-only" when:
1. Solar-thermal energy > threshold (default 0.001 kWh)
2. ASHP electricity ≤ threshold (default 0.016 kWh)
3. Immersion energy ≤ threshold (default 0.001 kWh)
4. All tank + ambient temperatures are finite (this row and previous)
5. No draw event (bottom-node temperature drop ≤ −0.25 °C)

Contiguous accepted intervals are grouped into windows.  Windows shorter
than `min_st_intervals` (default 4) are rejected.

### f_st calculation

For each accepted window:

```
dT_i = T_i[last_k] - T_i[first_k - 1]
energy_i = NODE_CAP_i × dT_i            (kJ, clipped ≥ 0)
f_st_window_i = energy_i / Σ_j energy_j
```

The overall `f_st` is the **median** across all valid windows (robust to
outliers), then normalised to sum to 1.

## Prerequisites

The `global_fit.json` from the Global_fitting pipeline is needed **only** for
the optional simulation plot.  The f_st calculation itself has no external
prior dependencies.

## Quick Start

```bash
# Basic run – produces st_fit.json + diagnostics:
python ST_fitting/run_st_fitting.py \
    --csv data/FullDS_Findhorn_5min.csv \
    --yaml column_mapping_5min.yaml

# With diagnostic plot:
python ST_fitting/run_st_fitting.py \
    --csv data/FullDS_Findhorn_5min.csv \
    --yaml column_mapping_5min.yaml \
    --plot

# Override thresholds:
python ST_fitting/run_st_fitting.py \
    --csv data/FullDS_Findhorn_5min.csv \
    --yaml column_mapping_5min.yaml \
    --st-on 0.005 --draw-delta -0.5
```

## Outputs

| File | Description |
|------|-------------|
| `output/st_fit.json` | Computed `f_st` + metadata and statistics |
| `diagnostics/st_intervals.csv` | Per-interval diagnostics with accept/reject reasons |
| `diagnostics/st_windows.csv` | Per-window summary: times, interval count, energy, per-node f_st |
| `output/plots/*.png` | Free forward simulation plot for selected ST window (with `--plot`) |

## Output JSON Schema

```json
{
  "f_st": [0.72, 0.15, 0.08, 0.05],
  "identification": {
    "n_windows_total": 150,
    "n_windows_valid": 142,
    "n_windows_skipped": 8,
    "total_st_energy_kwh": 45.6,
    "mean_st_per_window_kwh": 0.321,
    "mean_intervals_per_window": 6.3,
    "f_st_stats": {
      "nodes": ["bottom", "mid", "mid_hi", "top"],
      "mean": [0.68, 0.17, 0.09, 0.06],
      "median": [0.72, 0.15, 0.08, 0.05],
      "std": [0.12, 0.08, 0.05, 0.04],
      "n_windows": 142
    },
    "thresholds": { ... },
    "timestamp": "2026-04-08T..."
  }
}
```
