import pandas as pd
import numpy as np
from pathlib import Path

from . import data_loader
from .tank_model import NODE_CAP, TankParams

ROOT = Path(__file__).resolve().parent.parent

csv_path = ROOT / "FullDS_Findhorn.csv"
yaml_path = ROOT / "column_mapping.yaml"
train_frac = 0.7
dt_s = 1800.0

df = data_loader.load_and_clean(csv_path, yaml_path)

split_idx = int(len(df) * train_frac)
df_train = df.iloc[:split_idx].copy()

tank_cols = ["tank_bottom_c", "tank_mid_c", "tank_mid_hi_c", "tank_top_c"]

ashp_on = df_train["ashp_inst_kwh"].fillna(0) > 0.05
imm_off = df_train["imm_tot_inst_kwh"].fillna(0) < 0.01
st_low = df_train["st_kwh"].fillna(0) < 0.05 if "st_kwh" in df_train.columns else pd.Series(True, index=df.index)

T = df_train[tank_cols].values
finite_now = np.all(np.isfinite(T), axis=1)
finite_prev = np.roll(finite_now, 1)
finite_prev[0] = False

mask = ashp_on & imm_off & st_low & pd.Series(finite_now & finite_prev, index=df_train.index)

Q_back = pd.Series(np.nan, index=df_train.index)

idx = np.where(mask.values)[0]
ua_loss_default = TankParams().UA_loss

T_amb = df_train["t_amb_c"].fillna(df_train["t_amb_c"].median()).values

idx = np.where(mask.values)[0]
for k in idx:
    dT_sum = 0.0
    loss_sum = 0.0
    T_avg = 0.0
    for i in range(4):
        dT_sum += T[k, i] - T[k - 1, i]
        T_avg += T[k - 1, i]
        loss_sum += ua_loss_default[i] * (T[k - 1, i] - T_amb[k]) * dt_s
    T_avg /= 4.0

    Q_kJ = NODE_CAP * dT_sum + loss_sum  # kJ/K × K + kJ = kJ
    Q_back.iloc[k] = max(Q_kJ / 3600.0, 0.0)

df_train_reset = df_train.reset_index()  # This moves the index (time) back to a column
df_train_reset["Q_back_kwh"] = Q_back.values
out_df = df_train_reset[["time", "Q_back_kwh"]].dropna()
out_df.to_csv(ROOT / "ashp_only_intervals.csv", index=False)
print(mask.sum())