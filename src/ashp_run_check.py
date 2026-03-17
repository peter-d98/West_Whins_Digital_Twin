import pandas as pd
import numpy as np

df_raw = pd.read_csv(
    "data/Data_WestWhins_2023_2024_1min.csv",
    na_values=["#N/A"],
    low_memory=False,
)

# Rename to what the loader produces
df_raw.columns = df_raw.columns.str.strip()

# Compute diffs of cumulative columns
ashp_diff = df_raw["ASHP Elec [kWh]"].diff().fillna(0)
ashp_2min = ashp_diff.rolling(2, min_periods=1).sum()
imm_diff   = df_raw["Imm Elec [kWh]"].diff().fillna(0)
bkp_diff   = df_raw["Backup Imm Elec [kWh]"].diff().fillna(0)
imm_total  = imm_diff + bkp_diff

# How often is ASHP running (> 0.1 kWh/min)?
ashp_on_mask = ashp_2min > 0.10
imm_off_mask  = imm_total < 0.05

print("=== ASHP running minutes ===")
print(f"Total minutes with ASHP > 0.1 kWh: {ashp_on_mask.sum():,}")
print(f"Fraction of dataset: {ashp_on_mask.mean():.1%}")

print("\n=== Immersion stats ===")
print(f"imm_total describe:\n{imm_total.describe()}")
print(f"\nTop 10 imm_total values:\n{imm_total.value_counts().head(10)}")

print("\n=== ASHP on AND imm off ===")
both = ashp_on_mask & imm_off_mask
print(f"Minutes passing both: {both.sum():,}")

# Find longest consecutive run of (ashp_on & imm_off)
runs = []
in_run = False
start = 0
for i, val in enumerate(both.values):
    if val and not in_run:
        start = i
        in_run = True
    elif not val and in_run:
        runs.append(i - start)
        in_run = False
if in_run:
    runs.append(len(both) - start)

if runs:
    runs = sorted(runs, reverse=True)
    print(f"\nLongest 10 consecutive runs (ashp_on & imm_off):")
    for r in runs[:10]:
        print(f"  {r} minutes")
    print(f"\nRuns >= 15 min: {sum(1 for r in runs if r >= 15)}")
    print(f"Runs >= 10 min: {sum(1 for r in runs if r >= 10)}")
    print(f"Runs >= 5 min:  {sum(1 for r in runs if r >= 5)}")