# UA_fitting – Per-node UA_loss Fitting from Idle Tank Periods

## Overview

This tool detects periods when the DHW tank is "idle" (no heat input, no draws)
and uses the observed temperature decay to fit per-node `UA_loss` values that
characterise standing heat loss to the surrounding plant-room air.

The fitted values are compatible with `src.tank_model.TankParams.UA_loss` and
are saved in a JSON format ready for use in the hybrid global refinement
(to be implemented later).

## Physics

During idle periods the energy balance for each tank node simplifies to:

```
NODE_CAP × dT_i/dt ≈ −UA_loss_i × (T_i − T_amb) + (inter-node conduction)
```

By collecting many idle windows across the training dataset, we build a linear
system and solve for the 4 `UA_loss` values via least-squares regression
(optionally ridge-regularised).

## Quick Start

```bash
# Basic run – produces ua_fit.json + diagnostics CSV:
python UA_fitting/run_ua_fitting.py \
    --csv data/FullDS_Findhorn.csv \
    --yaml column_mapping.yaml

# With diagnostic plots (summer idle windows) and QC CSV:
python UA_fitting/run_ua_fitting.py \
    --csv data/FullDS_Findhorn.csv \
    --yaml column_mapping.yaml \
    --plot --qc-csv

# Override detection thresholds:
python UA_fitting/run_ua_fitting.py \
    --csv data/FullDS_Findhorn.csv \
    --yaml column_mapping.yaml \
    --ashp-off 0.05 --min-idle 3 --ridge 0.0
```

## Outputs

| File | Description |
|------|-------------|
| `output/ua_fit.json` | Fitted UA_loss array (4 values, bottom→top) + metadata |
| `diagnostics/idle_windows.csv` | All candidate windows with accept/reject reason |
| `output/qc_summary.csv` | Per-window RMSE and residual metrics (with `--qc-csv`) |
| `output/plots/*.png` | Measured vs predicted temperature plots (with `--plot`) |

## Output JSON Schema

```json
{
  "UA_loss": [0.00123, 0.00098, 0.00087, 0.00145],
  "metadata": {
    "n_windows": 42,
    "n_transitions": 186,
    "train_date_start": "2023-01-01 00:00:00",
    "train_date_end": "2024-06-15 23:30:00",
    "thresholds": { ... },
    "timestamp": "2026-03-21T12:00:00+00:00",
    "units": "kW/K (per node, bottom→top)",
    "node_cap_kj_per_k": 575.28,
    "dt_s": 1800.0
  }
}
```

## Using the Fitted UA in the Existing Pipeline

```python
import json
from src.tank_model import TankParams
import numpy as np

# Load fitted UA
with open("UA_fitting/output/ua_fit.json") as f:
    ua_result = json.load(f)

# Apply to TankParams
params = TankParams()
params.UA_loss = np.array(ua_result["UA_loss"])

# Use params in tank simulation or pass to joint refinement
```

## Programmatic API

```python
from src.data_loader import load_and_clean
from src.solar_thermal import compute_st_energy
from UA_fitting.config import UAConfig
from UA_fitting import detect_idle_windows, fit_ua, plot_qc

# Load data
df = load_and_clean("data/FullDS_Findhorn.csv", "column_mapping.yaml")
df["st_kwh"] = compute_st_energy(df)

# Configure (or use defaults)
cfg = UAConfig(min_idle_intervals=3, ridge_alpha=0.0)

# Detect → Fit → Plot
windows, diag_df = detect_idle_windows(df, cfg)
result = fit_ua(windows, cfg, output_dir=Path("UA_fitting/output"))
plot_qc(windows, result, cfg, output_dir=Path("UA_fitting/output/plots"))
```

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | (required) | Path to the plant CSV file |
| `--yaml` | (required) | Path to column_mapping.yaml |
| `--ashp-off` | 0.10 | ASHP-off threshold [kWh/interval] |
| `--st-off` | 0.05 | Solar-thermal-off threshold [kWh/interval] |
| `--imm-off` | 0.01 | Immersion-off threshold [kWh/interval] |
| `--min-idle` | 2 | Minimum consecutive idle intervals |
| `--draw-delta` | -2.0 | Bottom-node drop to flag a draw [°C] |
| `--min-windows` | 20 | Minimum idle windows required |
| `--ridge` | 1e-4 | Ridge regularisation weight (0 = off) |
| `--train-frac` | 0.7 | Fraction of data for training |
| `--plot` | off | Generate diagnostic plots |
| `--qc-csv` | off | Generate QC summary CSV |
| `--n-plot` | 3 | Number of windows to plot |
| `--summer-months` | 6 7 8 | Month numbers for summer window selection |

## Configurable Thresholds

All thresholds are defined in `config.py` as `UAConfig` dataclass attributes
and can be overridden via CLI arguments or by constructing a `UAConfig` object
directly.

## Notes

- **Training split**: uses the first 70% of the time-ordered data, matching the
  project convention (`train_frac=0.7`).
- **Negative UA allowed**: the optimiser can return negative values, which may
  indicate net energy gain from mixing effects or model bias.
- **Units**: UA_loss in kW/K, NODE_CAP in kJ/K, temperatures in °C, time in seconds.
- **30-minute data only** for now (set by `sampling_minutes=30`).
