# Stage‑1 – Data & Mapping 

This repository builds a **Stage‑1 digital twin** for a DHW system:
- 14 kW ATW heat pump; 550 L DHW tank with **four** nodes (bottom, mid, mid‑hi, top); 100 L buffer;
- Solar‑thermal array charging the DHW tank;
- Two 64 kW immersion heaters (backup/legionella).

**Stage‑1 uses solar‑thermal as measured** (from `ST Power [kW]` or derived from ST flow and ΔT); no forecasting or MPC yet.

## Input dataset

Provide a single **30‑minute** time‑series CSV with columns (or mappable equivalents):
- `Time`
- `Tank Bottom [°C]`, `Tank Mid [°C]`, `Tank Mid Hi [°C]`, `Tank Top [°C]`
- `ST Flow [L]`, `ST Flow T [°C]`, `ST Ret T [°C]`, `ST Power [kW]`
- `ASHP Elec [kWh]`, `ASHP Inst [kWh]`
- `Imm Tot Inst [kWh]` (and possibly `Imm Tot [kWh]`, `Backup Imm Elec [kWh]`)
- `PV Inst [kW]` (kept for completeness in Stage‑1)
- **`T_amb [C]`** (plant‑room ambient; used for tank losses and ASHP mapping)

These headers and cadence are consistent with the working dataset. Use `column_mapping.yaml` to map if they differ. 

## Known quirks to handle

- **Small negative `PV Inst [kW]`** values can occur at night; treat as expected artefacts.   
- Occasional **`#N/A` tokens** appear; parse as missing without breaking the pipeline.   
- **Cumulative meters** (e.g., `ASHP Elec [kWh]`) must be differenced to interval kWh with rollover repair.   
- If `ST Power [kW]` is missing, derive ST heat from flow × ΔT × ρ × cp × η_HX (η_HX identified). 

## Ambient usage

| Column | Meaning | Used for |
|---|---|---|
| `t_amb_c` | Plant-room temperature proxy (≈ outdoor air + 10 °C) | Tank UA heat-loss calculations |
| `t_out_c` | Estimated outdoor air temperature (`t_amb_c − 10.0`, derived by the loader) | ASHP performance mapping (capacity, power, COP) |

`t_out_c` is added automatically by the data loader whenever `t_amb_c` is present.

## Minimal expectations from the loader

- Parse `Time` (`%d/%m/%Y %H:%M`) and align to 30‑min grid.
- Standardise units and names via the mapping.
- Create interval energies, repair rollovers, and flag QC repairs.
- Clip impossible temperatures and log a QC message.
- Check node ordering (T_top ≥ T_mh ≥ T_mid ≥ T_bottom) for diagnostics.

## Out of scope in Stage‑1

- No forecasting (wind or ST) and no MPC.
- No pasteurisation scheduling (immersion energy still included in the balance).