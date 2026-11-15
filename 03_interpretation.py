"""Per-target importance, SHAP, correlation, parity and residual figures.

Uses the final XGBoost models from 02_models.py (tuned with grouped CV on the
literature data) and the grouped-CV out-of-fold predictions, so every plotted
prediction is for a composition the model did not see during training.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor

HERE = Path(__file__).parent
FIG, RES = HERE / "figures", HERE / "results"
COMP = ["C", "Si", "Mn", "P", "S", "Ni", "Cr", "Cu", "Mo"]
FEATURES = COMP + ["NT", "QT", "TT"]
TARGETS = {"Tensile": ("Tensile strength", "MPa"), "Hardness": ("Hardness", "HV")}
plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
                     "xtick.labelsize": 12, "ytick.labelsize": 12})

lit = pd.read_csv(HERE / "literature_medium_carbon.csv")
X = lit[FEATURES].astype(float)

rank_tables = []
fig_imp, axes_imp = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, (target, (label, unit)) in zip(axes_imp, TARGETS.items()):
    m = XGBRegressor(); m.load_model(str(RES / f"xgb_final_{target}.json"))
    gain = pd.Series(m.get_booster().get_score(importance_type="gain")).reindex(FEATURES).fillna(0)
    gain_pct = 100 * gain / gain.sum()
    sv = shap.TreeExplainer(m).shap_values(X)
    mean_abs = pd.Series(np.abs(sv).mean(0), index=FEATURES)
    corr = X.corrwith(lit[target])
    t = pd.DataFrame({"Gain (%)": gain_pct.round(1), "Gain rank": gain_pct.rank(ascending=False).astype(int),
                      "Mean |SHAP|": mean_abs.round(1), "Unit": unit,
                      "SHAP rank": mean_abs.rank(ascending=False).astype(int),
                      "Pearson r": corr.round(2)}).sort_values("SHAP rank")
    t.insert(0, "Target", target); rank_tables.append(t)

    g = gain_pct.sort_values()
    ax.barh(g.index, g.values, color="#4878a8")
    for y, v in enumerate(g.values):
        ax.text(v + 0.5, y, f"{v:.1f}", va="center", fontsize=11)
    ax.set_xlabel("Share of total gain (%)"); ax.set_title(f"({'ab'[list(TARGETS).index(target)]}) {label}")
    ax.set_xlim(0, g.max() * 1.18)

    plt.figure()
    shap.summary_plot(sv, X, show=False, plot_size=(8, 6))
    plt.xlabel(f"SHAP value (impact on predicted {label.lower()}, {unit})")
    plt.tight_layout(); plt.savefig(FIG / f"fig_shap_summary_{target}.png", dpi=300); plt.close()

    for feat in ["C", "Cr", "TT"]:
        plt.figure()
        shap.dependence_plot(feat, sv, X, interaction_index=None, show=False)
        plt.ylabel(f"SHAP value for {feat} ({unit})")
        plt.xlabel(f"{feat} ({'°C' if feat in ('NT', 'QT', 'TT') else 'wt.%'})")
        plt.tight_layout(); plt.savefig(FIG / f"fig_shap_dependence_{target}_{feat}.png", dpi=300); plt.close()

fig_imp.tight_layout(); fig_imp.savefig(FIG / "fig_gain_importance_per_target.png", dpi=300); plt.close(fig_imp)
pd.concat(rank_tables).to_csv(RES / "importance_shap_correlation_ranks.csv")

# Feature-target correlation heatmap (inputs vs both targets only)
corr = pd.concat([X, lit[list(TARGETS)]], axis=1).corr().loc[FEATURES, list(TARGETS)]
fig, ax = plt.subplots(figsize=(5, 7))
im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(2), ["Tensile", "Hardness"]); ax.set_yticks(range(len(FEATURES)), FEATURES)
for i in range(len(FEATURES)):
    for j in range(2):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=11)
fig.colorbar(im, label="Pearson r"); fig.tight_layout()
fig.savefig(FIG / "fig_feature_target_correlation.png", dpi=300); plt.close(fig)

# Parity and residual plots from grouped-CV out-of-fold predictions (XGBoost)
oof = pd.read_csv(RES / "grouped_cv_out_of_fold_predictions.csv")
oof = oof[oof.Model == "XGBoost"]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for col, (target, (label, unit)) in enumerate(TARGETS.items()):
    d = oof[oof.Target == target]
    ax = axes[0, col]
    lo, hi = d[["Actual", "Predicted"]].min().min(), d[["Actual", "Predicted"]].max().max()
    ax.scatter(d.Actual, d.Predicted, s=22, alpha=0.7, color="#4878a8")
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y = x")
    ax.set_xlabel(f"Measured {label.lower()} ({unit})"); ax.set_ylabel(f"Predicted ({unit})")
    ax.set_title(f"{label}: grouped-CV out-of-fold predictions"); ax.legend()
    r = d.Actual - d.Predicted
    ax = axes[1, col]
    ax.hist(r, bins=30, color="#4878a8", edgecolor="white")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel(f"Residual, measured − predicted ({unit})"); ax.set_ylabel("Count")
    ax.set_title(f"mean = {r.mean():.1f}, sd = {r.std():.1f} {unit}")
fig.tight_layout(); fig.savefig(FIG / "fig_parity_residuals_groupedCV.png", dpi=300); plt.close(fig)

print(pd.concat(rank_tables).to_string())
