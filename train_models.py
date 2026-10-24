# =====================================================================
#  train_models.py  -  compares 6 regression models on the microstrip data
#  Run:  python train_models.py
#  Needs: pip install numpy pandas scikit-learn matplotlib
#  Keep this file in the same folder as the "data" folder.
# =====================================================================
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # lets plots be saved without a window
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------
# STEP A: describe the two problems (file names, inputs, output)
# ---------------------------------------------------------------------
PROBLEMS = {
    "microstrip_z0": {"inputs": ["W_over_h", "eps_r"], "output": "Z0_ohm"},
    "patch_fr":      {"inputs": ["W_mm", "L_mm", "h_mm", "eps_r"], "output": "fr_MHz"},
}

# ---------------------------------------------------------------------
# STEP B: the six models. Each is a function that builds a fresh model.
# 'param' is the setting we tune on the validation set.
# Inputs are scaled to similar ranges (StandardScaler); for the models
# that need it, the output is scaled too (TransformedTargetRegressor).
# ---------------------------------------------------------------------
def scaled_target(model):
    return TransformedTargetRegressor(regressor=model, transformer=StandardScaler())

MODELS = {
    "Linear regression": (lambda p: make_pipeline(StandardScaler(), LinearRegression()), [None]),
    "Polynomial regression": (lambda p: make_pipeline(StandardScaler(), PolynomialFeatures(p), LinearRegression()),
                              [2, 3, 4, 5, 6]),                      # tune: degree
    "Decision tree": (lambda p: DecisionTreeRegressor(max_depth=p, random_state=0),
                      [4, 8, 12, 16, None]),                         # tune: depth
    "Random forest": (lambda p: RandomForestRegressor(n_estimators=200, max_depth=p, random_state=0, n_jobs=-1),
                      [8, 12, None]),                                # tune: depth
    "SVR (RBF)": (lambda p: scaled_target(make_pipeline(StandardScaler(), SVR(kernel="rbf", C=p, epsilon=0.01))),
                  [1, 10, 100]),                                     # tune: C
    "Neural network": (lambda p: scaled_target(make_pipeline(StandardScaler(),
                          MLPRegressor(hidden_layer_sizes=p, max_iter=3000, random_state=0))),
                       [(32, 32), (64, 64), (64, 64, 32)]),          # tune: layer sizes
}

# ---------------------------------------------------------------------
# STEP C: error measures (all computed on data the model has NOT seen)
# ---------------------------------------------------------------------
def errors(y_true, y_pred):
    abs_err = np.abs(y_true - y_pred)
    pct_err = abs_err / np.abs(y_true) * 100
    return {
        "MAE": mean_absolute_error(y_true, y_pred),            # average error
        "Max error": abs_err.max(),                            # worst case
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "Mean %err": pct_err.mean(),
        "Max %err": pct_err.max(),
    }

# ---------------------------------------------------------------------
# STEP D: main loop - for each problem, train, tune, test, print, plot
# ---------------------------------------------------------------------
for name, cfg in PROBLEMS.items():
    X_cols, y_col = cfg["inputs"], cfg["output"]
    train = pd.read_csv(f"data/{name}_train.csv")
    val   = pd.read_csv(f"data/{name}_val.csv")
    test  = pd.read_csv(f"data/{name}_test.csv")
    Xtr, ytr = train[X_cols], train[y_col]
    Xva, yva = val[X_cols],   val[y_col]
    Xte, yte = test[X_cols],  test[y_col]

    print(f"\n=== {name}:  predicting {y_col} from {X_cols} ===")
    rows, fitted, preds = [], {}, {}

    for mname, (build, options) in MODELS.items():
        # 1) tune: try each option, keep the one with the lowest validation error
        best_p, best_mae = None, np.inf
        for p in options:
            m = build(p).fit(Xtr, ytr)
            mae = mean_absolute_error(yva, m.predict(Xva))
            if mae < best_mae:
                best_p, best_mae = p, mae
        # 2) final training with the best setting, timing the training
        t0 = time.time(); model = build(best_p).fit(Xtr, ytr); train_s = time.time() - t0
        # 3) test on unseen data, timing the prediction
        t0 = time.time(); y_pred = model.predict(Xte); pred_us = (time.time() - t0) / len(Xte) * 1e6
        r = errors(yte.values, y_pred)
        r.update({"Model": mname, "Setting": str(best_p), "Train time (s)": train_s, "Predict (us/sample)": pred_us})
        rows.append(r); fitted[mname] = model; preds[mname] = y_pred

    table = pd.DataFrame(rows).set_index("Model")
    cols = ["Setting", "MAE", "Max error", "RMSE", "R2", "Mean %err", "Max %err", "Train time (s)", "Predict (us/sample)"]
    table = table[cols]
    print(table.round(4).to_string())
    table.round(5).to_csv(f"results_{name}.csv")

    # ---- plot 1: bar chart of test errors (lower is better)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].barh(table.index, table["MAE"], color="#1D4E89");        ax[0].set_title("Average error (MAE)")
    ax[1].barh(table.index, table["Max error"], color="#B45309");  ax[1].set_title("Maximum error")
    for a in ax: a.invert_yaxis(); a.set_xscale("log"); a.grid(alpha=.3)
    fig.suptitle(f"{name}: test-set error (log scale)"); fig.tight_layout()
    fig.savefig(f"errors_{name}.png", dpi=150); plt.close(fig)

    # ---- plot 2: actual vs predicted on the test set (points on the diagonal = perfect)
    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    for a, (mname, yp) in zip(axs.ravel(), preds.items()):
        a.scatter(yte, yp, s=4, alpha=.5); lo, hi = yte.min(), yte.max()
        a.plot([lo, hi], [lo, hi], "r--", lw=1); a.set_title(mname, fontsize=10)
        a.set_xlabel("actual"); a.set_ylabel("predicted")
    fig.suptitle(f"{name}: predicted vs actual (test set)"); fig.tight_layout()
    fig.savefig(f"predicted_vs_actual_{name}.png", dpi=150); plt.close(fig)

    # ---- plot 3 (microstrip only): curve of Z0 vs W/h, like the paper's figures
    if name == "microstrip_z0":
        wh = np.linspace(1, 10, 200); er = 6.0
        eeff = (er + 1) / 2 + (er - 1) / 2 / np.sqrt(1 + 12 / wh)
        actual = 120 * np.pi / (np.sqrt(eeff) * (wh + 1.393 + 2 / 3 * np.log(wh + 1.444)))
        grid = pd.DataFrame({"W_over_h": wh, "eps_r": er})
        plt.figure(figsize=(8, 5)); plt.plot(wh, actual, "k", lw=3, label="Actual (formula)")
        for mname, model in fitted.items(): plt.plot(wh, model.predict(grid), "--", label=mname)
        plt.xlabel("W/h"); plt.ylabel("Z0 (ohm)"); plt.title(f"Z0 vs W/h at eps_r = {er}")
        plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig("curve_microstrip.png", dpi=150); plt.close()

    # ---- optional: check against your own ADS/HFSS results (step 4)
    # Create data/ads_check.csv with the SAME columns as the microstrip files:
    # W_over_h,eps_r,Z0_ohm   (Z0 values read from ADS LineCalc or HFSS)
    if name == "microstrip_z0":
        try:
            ads = pd.read_csv("data/ads_check.csv")
            print("\n--- Check against ADS/HFSS values (mean % error) ---")
            for mname, model in fitted.items():
                e = errors(ads[y_col].values, model.predict(ads[X_cols]))
                print(f"{mname:22s} mean %err = {e['Mean %err']:.2f}   max %err = {e['Max %err']:.2f}")
        except FileNotFoundError:
            print("\n(no data/ads_check.csv found - skipping the ADS/HFSS check)")
