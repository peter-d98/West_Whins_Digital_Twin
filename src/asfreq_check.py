import pandas as pd
df = pd.read_csv("data/Data_WestWhins_2023_2024_1min.csv", na_values=["#N/A"])
df["Time"] = pd.to_datetime(df["Time"], format="mixed")
df = df.sort_values("Time").drop_duplicates("Time").set_index("Time")
df2 = df.asfreq("1min")
print(f"Before asfreq: {len(df)} rows")
print(f"After asfreq:  {len(df2)} rows")
print(f"NaN rows introduced: {len(df2) - len(df)}")
ashp_diff = df2["ASHP Elec [kWh]"].diff().fillna(0)
ashp_on = ashp_diff > 0.05
# Consecutive runs
import numpy as np
mask = ashp_on.values
runs = []
in_run, start = False, 0
for i, v in enumerate(mask):
    if v and not in_run: start, in_run = i, True
    elif not v and in_run: runs.append(i - start); in_run = False
if in_run: runs.append(len(mask) - start)
runs.sort(reverse=True)
print(f"\nRuns >= 5 min:  {sum(1 for r in runs if r >= 5)}")
print(f"Runs >= 10 min: {sum(1 for r in runs if r >= 10)}")
print(f"Longest 10: {runs[:10]}")