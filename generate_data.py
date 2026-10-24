# Generates the datasets for the microstrip ML project.
# Run:  python generate_data.py      (needs numpy and pandas)
import numpy as np
import pandas as pd
import os
os.makedirs("data", exist_ok=True)   # create the data folder if missing

rng = np.random.default_rng(42)      # fixed seed -> same data every time
N = 10000                            # samples per dataset
C = 3e8                              # speed of light (m/s)

def eps_eff(wh, er):                 # effective permittivity
    return (er + 1) / 2 + (er - 1) / 2 / np.sqrt(1 + 12 / wh)

# ---- Dataset 1: microstrip line impedance Z0 (Gupta & Garg, valid W/h > 1)
wh = rng.uniform(1, 10, N)           # W/h ratio
er = rng.uniform(2.2, 12, N)         # relative permittivity
ee = eps_eff(wh, er)
z0 = 120 * np.pi / (np.sqrt(ee) * (wh + 1.393 + 2 / 3 * np.log(wh + 1.444)))
line = pd.DataFrame({"W_over_h": wh, "eps_r": er, "Z0_ohm": z0})

# ---- Dataset 2: rectangular patch resonant frequency fr (Singh)
h = rng.uniform(0.5, 3.0, N)         # substrate height (mm)
wh = rng.uniform(1, 10, N)           # W/h ratio, so W = wh * h
W = wh * h                           # patch width (mm)
L = rng.uniform(5, 15, N)            # patch length (mm)
er = rng.uniform(2.2, 12, N)
ee = eps_eff(wh, er)
dL = 0.412 * h * ((ee + 0.3) / (ee - 0.258)) * ((wh + 0.264) / (wh + 0.8))   # mm
fr = C / (2 * np.sqrt(ee) * (L + 2 * dL) * 1e-3) / 1e6                       # MHz
patch = pd.DataFrame({"W_mm": W, "L_mm": L, "h_mm": h, "eps_r": er, "fr_MHz": fr})

# ---- 70 / 15 / 15 split (train / validation / test)
def save(df, name):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    a, b = int(0.70 * len(df)), int(0.85 * len(df))
    df.iloc[:a].to_csv(f"data/{name}_train.csv", index=False, float_format="%.5f")
    df.iloc[a:b].to_csv(f"data/{name}_val.csv", index=False, float_format="%.5f")
    df.iloc[b:].to_csv(f"data/{name}_test.csv", index=False, float_format="%.5f")

save(line, "microstrip_z0")
save(patch, "patch_fr")
print(line.describe().round(2)); print(patch.describe().round(2))
