# Global Fitting Pipeline

Fits the remaining free tank parameters — `UA_adj` (3 inter-node coupling
values) and optionally `UA_loss[0]` (bottom-node standing loss) — using
frozen priors from the UA and ASHP fitting pipelines.

## Prerequisites

Run both upstream pipelines first:

```bash
python UA_fitting/run_ua_fitting.py --csv data/FullDS_Findhorn.csv --yaml column_mapping.yaml
python ASHP_fitting/run_ashp_fitting.py --csv data/FullDS_Findhorn.csv --yaml column_mapping.yaml
```

## Usage

```bash
python Global_fitting/run_global_fitting.py \
    --csv data/FullDS_Findhorn.csv \
    --yaml column_mapping.yaml \
    --plot
```

### CLI options

| Flag | Description |
|------|-------------|
| `--csv` | Path to the 30-min cleaned CSV |
| `--yaml` | Path to column_mapping.yaml |
| `--ua-json` | Override path to ua_fit.json |
| `--ashp-json` | Override path to ashp_fit.json |
| `--train-frac` | Training fraction (default 0.7) |
| `--max-nfev` | Max function evaluations (default 500) |
| `--freeze-ua-loss-bottom` | Freeze UA_loss[0] at its prior value |
| `--plot` | Generate prediction-error diagnostic plots |

## Inputs

| File | Source |
|------|--------|
| `UA_fitting/output/ua_fit.json` | `UA_loss` array (4 values) |
| `ASHP_fitting/output/ashp_fit.json` | `ASHPParams` (a, b arrays) |
| 30-min cleaned CSV + column_mapping.yaml | Tank temperatures, heat inputs |

## Outputs

| File | Contents |
|------|----------|
| `Global_fitting/output/global_fit.json` | Fitted `UA_loss`, `UA_adj`, `f_st`, `f_ashp`, `f_imm`, identification metrics |
| `Global_fitting/output/plots/` | Prediction-error time-series plots (train and validation) |

## Method

The fitting uses **one-step-ahead prediction**: at each interval *k*, the
measured temperatures T[k−1] are used as the initial state, `tank_step()` is
called once to predict T[k], and the residual is T_pred[k] − T_meas[k].

Physical heat-distribution fractions are fixed (not optimised):
- `f_st = [1, 0, 0, 0]` — ST coil in bottom node
- `f_ashp = [0, 0, 0, 1]` — ASHP returns to top node
- `f_imm = [0, 1, 0, 0]` — immersion in mid node
