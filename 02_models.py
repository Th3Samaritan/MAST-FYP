"""Retrain RF, ANN and XGBoost per target on literature data only.

Protocols (all tuning happens inside the training portion only):
  A  random 80/20 split, 5-fold inner grid search  (comparable to the original paper)
  B  nested random 5-fold CV                        (mean +/- sd across folds)
  C  nested composition-grouped 5-fold CV           (GroupKFold by steel composition)
Hold-out: final model tuned with grouped CV on all literature rows, then applied
once to the in-house sample, which is never seen during training or tuning.
All scaling lives inside sklearn Pipelines, so it is fitted on training folds only.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
RES = HERE / "results"
SEED = 42
COMP = ["C", "Si", "Mn", "P", "S", "Ni", "Cr", "Cu", "Mo"]
FEATURES = COMP + ["NT", "QT", "TT"]
TARGETS = {"Tensile": "MPa", "Hardness": "HV"}

lit = pd.read_csv(HERE / "literature_medium_carbon.csv")
inhouse = pd.read_csv(HERE / "inhouse_sample.csv")
X, X_in = lit[FEATURES].astype(float), inhouse[FEATURES].astype(float)
groups = lit.groupby(COMP).ngroup().to_numpy()


def models():
    return {
        "Random Forest": (
            RandomForestRegressor(random_state=SEED, n_jobs=-1),
            {"n_estimators": [300], "max_depth": [None, 8], "min_samples_leaf": [1, 3],
             "max_features": [1.0, 0.5]}),
        "ANN": (
            TransformedTargetRegressor(
                regressor=Pipeline([("scale", StandardScaler()),
                                    ("mlp", MLPRegressor(max_iter=5000, early_stopping=False,
                                                         random_state=SEED))]),
                transformer=StandardScaler()),
            {"regressor__mlp__hidden_layer_sizes": [(32,), (64, 32)],
             "regressor__mlp__alpha": [1e-3, 1e-1]}),
        "XGBoost": (
            XGBRegressor(random_state=SEED, tree_method="hist", n_jobs=-1),
            {"n_estimators": [300, 600], "max_depth": [3, 5], "learning_rate": [0.03, 0.1],
             "subsample": [0.8], "colsample_bytree": [0.8]}),
    }


def metrics(y, p):
    return {"R2": r2_score(y, p), "MAE": mean_absolute_error(y, p),
            "RMSE": float(np.sqrt(mean_squared_error(y, p)))}


def tune(est, grid, Xt, yt, inner_cv, inner_groups=None):
    gs = GridSearchCV(est, grid, cv=inner_cv, scoring="neg_mean_absolute_error", n_jobs=-1)
    gs.fit(Xt, yt, groups=inner_groups)
    return gs


rows, oof_rows, holdout_rows, params = [], [], [], {}
for target, unit in TARGETS.items():
    y = lit[target].astype(float)
    for name, (est, grid) in models().items():
        # Protocol A: random 80/20, as in the original manuscript
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
        gs = tune(est, grid, Xtr, ytr, KFold(5, shuffle=True, random_state=SEED))
        m = metrics(yte, gs.predict(Xte))
        rows.append({"Target": target, "Unit": unit, "Model": name, "Protocol": "A: random 80/20 test set",
                     **{k: v for k, v in m.items()}, **{f"{k}_sd": np.nan for k in m}})

        # Protocols B and C: nested CV
        for proto, outer, use_groups in [
            ("B: random 5-fold CV", KFold(5, shuffle=True, random_state=SEED), False),
            ("C: grouped 5-fold CV (by composition)", GroupKFold(5), True),
        ]:
            fold_m, oof = [], np.zeros(len(y))
            split = outer.split(X, y, groups if use_groups else None)
            for tr, te in split:
                inner = GroupKFold(5) if use_groups else KFold(5, shuffle=True, random_state=SEED)
                g = tune(est, grid, X.iloc[tr], y.iloc[tr], inner, groups[tr] if use_groups else None)
                oof[te] = g.predict(X.iloc[te])
                fold_m.append(metrics(y.iloc[te], oof[te]))
            fm = pd.DataFrame(fold_m)
            rows.append({"Target": target, "Unit": unit, "Model": name, "Protocol": proto,
                         **fm.mean().to_dict(), **{f"{k}_sd": v for k, v in fm.std(ddof=1).items()}})
            oof_rows.append(pd.DataFrame({"Target": target, "Model": name, "Protocol": proto[0],
                                          "Actual": y, "Predicted": oof, "Group": groups}))

        # Final model: tuned with grouped CV on all literature data; predict in-house once
        g = tune(est, grid, X, y, GroupKFold(5), groups)
        params[f"{target} | {name}"] = g.best_params_
        pred = float(g.predict(X_in)[0])
        act = float(inhouse[target].iloc[0])
        holdout_rows.append({"Target": target, "Unit": unit, "Model": name, "Actual": act,
                             "Predicted": round(pred, 1), "Abs error": round(abs(pred - act), 1),
                             "Error (%)": round(abs(pred - act) / act * 100, 2)})
        if name == "XGBoost":
            g.best_estimator_.save_model(str(RES / f"xgb_final_{target}.json"))
        print(f"done {target} / {name}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(RES / "model_performance_all_protocols.csv", index=False)
oof_all = pd.concat(oof_rows)
oof_all.to_csv(RES / "cv_out_of_fold_predictions.csv", index=False)
oof_all[oof_all.Protocol == "C"].drop(columns="Protocol").to_csv(RES / "grouped_cv_out_of_fold_predictions.csv", index=False)
pd.DataFrame(holdout_rows).to_csv(RES / "inhouse_holdout_predictions.csv", index=False)
(RES / "final_hyperparameters.json").write_text(json.dumps(params, indent=2, default=str))

fmt = res.copy()
for k in ["R2", "MAE", "RMSE"]:
    dec = 3 if k == "R2" else 1
    fmt[k] = [f"{v:.{dec}f}" + ("" if np.isnan(s) else f" ± {s:.{dec}f}") for v, s in zip(res[k], res[f"{k}_sd"])]
fmt = fmt[["Target", "Unit", "Model", "Protocol", "R2", "MAE", "RMSE"]]
fmt.to_csv(RES / "table1_performance_formatted.csv", index=False)
pd.set_option("display.width", 200)
print(fmt.to_string(index=False))
print(pd.DataFrame(holdout_rows).to_string(index=False))
print(json.dumps(params, indent=1, default=str))
