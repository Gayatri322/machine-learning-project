# =====================================================================
#  extrapolation_test.py - can the models predict W/h values they never saw?
#  Train on W/h 1-7, tune on W/h 7-8, test on W/h 8-10 (unseen range).
#  Run:  python extrapolation_test.py   (same folder as the "data" folder)
# =====================================================================
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

def scaled_target(m): return TransformedTargetRegressor(regressor=m, transformer=StandardScaler())
MODELS = {
    "Linear regression": (lambda p: make_pipeline(StandardScaler(), LinearRegression()), [None]),
    "Polynomial regression": (lambda p: make_pipeline(StandardScaler(), PolynomialFeatures(p), LinearRegression()), [2, 3, 4, 5, 6]),
    "Decision tree": (lambda p: DecisionTreeRegressor(max_depth=p, random_state=0), [4, 8, 12, 16, None]),
    "Random forest": (lambda p: RandomForestRegressor(n_estimators=200, max_depth=p, random_state=0, n_jobs=-1), [8, 12, None]),
    "SVR (RBF)": (lambda p: scaled_target(make_pipeline(StandardScaler(), SVR(kernel="rbf", C=p, epsilon=0.01))), [1, 10, 100]),
    "Neural network": (lambda p: scaled_target(make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=p, max_iter=3000, random_state=0))), [(32, 32), (64, 64), (64, 64, 32)]),
}
PROBLEMS = {"microstrip_z0": (["W_over_h", "eps_r"], "Z0_ohm"),
            "patch_fr": (["W_mm", "L_mm", "h_mm", "eps_r"], "fr_MHz")}

for name, (X_cols, y_col) in PROBLEMS.items():
    df = pd.concat([pd.read_csv(f"data/{name}_{s}.csv") for s in ("train", "val", "test")], ignore_index=True)
    wh = df["W_over_h"] if "W_over_h" in df else df["W_mm"] / df["h_mm"]     # W/h of every row
    tr, va, te = df[wh < 7], df[(wh >= 7) & (wh < 8)], df[wh >= 8]
    print(f"\n=== {name}: train {len(tr)}, val {len(va)}, test (W/h 8-10) {len(te)} ===")
    rows = []
    for mname, (build, options) in MODELS.items():
        best_p = min(options, key=lambda p: mean_absolute_error(va[y_col], build(p).fit(tr[X_cols], tr[y_col]).predict(va[X_cols])))
        pred = build(best_p).fit(tr[X_cols], tr[y_col]).predict(te[X_cols])
        ae = np.abs(te[y_col].values - pred); pe = ae / np.abs(te[y_col].values) * 100
        rows.append({"Model": mname, "Setting": str(best_p), "MAE": ae.mean(), "Max error": ae.max(),
                     "Mean %err": pe.mean(), "Max %err": pe.max()})
    t = pd.DataFrame(rows).set_index("Model"); print(t.round(4).to_string())
    t.round(5).to_csv(f"results_extrapolation_{name}.csv")
