# ASHP_fitting – ASHP Performance-Map Identification from 30-Minute Data

## Overview

This tool identifies clean ASHP-only intervals in 30-minute plant data,
back-calculates the condenser heat delivered to the DHW tank using the 4-node
energy balance, and fits bilinear ASHP capacity and power maps.

The fitted maps are compatible with `src.ashp_model.ASHPParams` and can be
supplied to `run_stage1.py --ashp-params` to replace or validate the default
ASHP identification.

## Physics

### ASHP-only interval detection

An interval qualifies as "ASHP-only" when:
1. ASHP electricity > threshold (default 0.013 kWh)
2. Solar-thermal energy ≤ threshold (default 0.05 kWh)
3. Immersion energy ≤ threshold (default 0.01 kWh)
4. All tank + ambient temperatures are finite (this row and previous)
5. No draw event (bottom-node temperature drop ≤ −1.0 °C)

**Draw rejection is interval-only**: a draw in one interval removes only that
interval, not the entire surrounding window.

### Back-calculation

For each accepted interval *k*:

```
Q_ashp_kJ = NODE_CAP × Σᵢ (T_i[k] − T_i[k−1])
            + Σᵢ UA_loss_i × (T_i[k−1] − T_amb[k]) × dt_s
```

where:
- `NODE_CAP ≈ 575.3 kJ/K` (per node, 137.5 L water)
- `UA_loss` is loaded from `UA_fitting/output/ua_fit.json` (required prior)
- `dt_s = 1800 s` for 30-minute data

Only intervals with `Q_back > 0` are used for fitting.

### Map fitting

Bilinear performance maps:
```
Q_cond [kW] = a₀ + a₁·T_out + a₂·T_sink + a₃·T_out·T_sink
P_elec [kW] = b₀ + b₁·T_out + b₂·T_sink + b₃·T_out·T_sink
```

Fitting uses OLS initialisation followed by robust `least_squares` with
`soft_l1` loss (via `src.ashp_model.fit_ashp_maps`).

## Prerequisites

The UA fitting pipeline must be run first to produce `UA_fitting/output/ua_fit.json`:

```bash
python UA_fitting/run_ua_fitting.py \
    --csv data/FullDS_Findhorn.csv \
    --yaml column_mapping.yaml
```

## Quick Start

```bash
# Basic run – produces ashp_fit.json + diagnostics:
python ASHP_fitting/run_ashp_fitting.py \
    --csv data/FullDS_Findhorn.csv \
    --yaml column_mapping.yaml

# With diagnostic plots:
python ASHP_fitting/run_ashp_fitting.py \
    --csv data/FullDS_Findhorn.csv \
    --yaml column_mapping.yaml \
    --plot

# Override detection thresholds:
python ASHP_fitting/run_ashp_fitting.py \
    --csv data/FullDS_Findhorn.csv \
    --yaml column_mapping.yaml \
    --ashp-off 0.05 --draw-delta -2.0

# Custom UA fit path:
python ASHP_fitting/run_ashp_fitting.py \
    --csv data/FullDS_Findhorn.csv \
    --yaml column_mapping.yaml \
    --ua-json path/to/ua_fit.json
```

## Outputs

| File | Description |
|------|-------------|
| `output/ashp_fit.json` | Fitted ASHP map coefficients (a, b) + metadata |
| `diagnostics/ashp_intervals.csv` | Per-interval diagnostics with accept/reject reasons |
| `diagnostics/backcalc_details.csv` | Back-calculated Q, P, T_out, T_sink per accepted interval |
| `output/plots/*.png` | Node temperature plots for longest ASHP windows (with `--plot`) |

## Output JSON Schema

```json
{
  "ashp": {
    "a": [float, float, float, float],
    "b": [float, float, float, float]
  },
  "identification": {
    "n_intervals_accepted": 500,
    "n_intervals_q_positive": 480,
    "n_intervals_q_rejected": 20,
    "n_windows": 120,
    "mean_back_cop": 2.5,
    "median_back_cop": 2.4,
    "cop_at_A-3/W55": 1.8,
    "cop_at_A7/W55": 3.2,
    "cop_at_A-3/W35": 2.1,
    "thresholds": { ... },
    "ua_loss_used": [float, float, float, float],
    "timestamp": "2026-03-22T12:00:00+00:00"
  }
}
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | (required) | Path to the plant CSV file |
| `--yaml` | (required) | Path to column_mapping.yaml |
| `--ua-json` | `UA_fitting/output/ua_fit.json` | Path to UA priors JSON |
| `--ashp-off` | 0.013 | ASHP-off threshold [kWh/interval] |
| `--st-off` | 0.05 | Solar-thermal-off threshold [kWh/interval] |
| `--imm-off` | 0.01 | Immersion-off threshold [kWh/interval] |
| `--draw-delta` | −1.0 | Bottom-node draw threshold [°C] |
| `--min-intervals` | 1 | Minimum consecutive ASHP-only intervals per window |
| `--train-frac` | 0.7 | Fraction of dataset used for fitting |
| `--high-load-filter` | off | Enable 75th-percentile power filter |
| `--plot` | off | Generate diagnostic window plots |
| `--n-plot` | 5 | Number of windows to plot |

## Known Limitations

1. **30-minute cadence**: At 30-min resolution, short ASHP runs may span
   only 1–2 intervals, giving noisy back-calculations.
2. **UA_loss priors**: The back-calculation quality depends on the accuracy
   of the fitted `UA_loss` values.  A negative `UA_loss[0]` (bottom node)
   is common and reflects absorbed mixing/modelling artefacts.
3. **No draw-off heat accounting**: During a draw, cold mains water enters
   the tank.  We reject these intervals entirely rather than modelling the
   draw energy.
4. **High-load filter disabled**: By default, all valid ASHP intervals are
   used for fitting.  Enable `--high-load-filter` to restrict to >75th
   percentile power intervals (steady-state operation).
5. **Solar-thermal leakage**: ST energy is derived from measured power or
   flow × ΔT.  If the ST measurement is noisy, some ST contamination may
   remain in nominally "ST-off" intervals.

## Relation to Other Modules

- **`UA_fitting/`**: Produces the `ua_fit.json` priors required by this module.
- **`run_ashp_1min.py`**: Alternative ASHP identification from 1-minute data
  (run-based, not interval-based).
- **`run_stage1.py`**: Can consume the output `ashp_fit.json` via
  `--ashp-params` to bypass its internal ASHP identification.
