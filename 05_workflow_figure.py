"""Pipeline overview figure (Reviewer 3, comment 2).

Shows the ML branch and the phase-diagram branch as parallel modules that
meet only in the Streamlit interface.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).parent
fig, ax = plt.subplots(figsize=(13, 8.2))
ax.set_xlim(0, 13); ax.set_ylim(0, 8.2); ax.axis("off")

C_DATA, C_ML, C_EXP, C_PD, C_APP = "#d6eaf8", "#d5f5e3", "#fdebd0", "#e8daef", "#fcf3cf"


def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                fc=color, ec="#34495e", lw=1.4))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=11.5, weight="bold")
    ax.text(x + w / 2, y + h - 0.6, body, ha="center", va="top", fontsize=9.6, linespacing=1.35)


def arrow(p, q, style="-|>", ls="-", color="#34495e"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=16, lw=1.5,
                                 color=color, linestyle=ls))


# Row 1: data
box(0.3, 6.1, 3.6, 1.8, "Literature data", "NIMS steel database\n(compiled by Xiong et al.)\n360 records → 357 with\n0.30–0.60 wt.% C", C_DATA)
box(4.6, 6.1, 3.8, 1.8, "Pre-processing", "12 inputs: C, Si, Mn, P, S, Ni,\nCr, Cu, Mo, NT, QT, TT\nScaling fitted on training\nfolds only (Pipeline)", C_DATA)
box(9.1, 6.1, 3.6, 1.8, "In-house experiment", "0.40 wt.% C steel: OES,\nnormalising 860 °C / 1 h,\ntensile + Vickers hardness\n(never used in training)", C_EXP)

# Row 2: ML
box(0.3, 3.4, 3.6, 2.1, "Model training", "RF, ANN, XGBoost\none model per target\n(tensile, hardness)\nnested grid search", C_ML)
box(4.6, 3.4, 3.8, 2.1, "Validation", "Random 80/20 test\nRandom 5-fold CV\nComposition-grouped\n5-fold CV (GroupKFold)", C_ML)
box(9.1, 3.4, 3.6, 2.1, "Final models + external test", "Re-tuned on all 357 records;\npredict the in-house steel\nonce; check domain of\napplicability", C_EXP)

# Row 3: interpretation, phase diagram, app
box(0.3, 0.5, 3.6, 2.3, "Interpretation", "Gain importance and\nSHAP per target\n(statistical attribution,\ndiscussed against\nmetallurgical literature)", C_ML)
box(4.6, 0.5, 3.8, 2.3, "Fe–Fe$_3$C reference module", "Binary, equilibrium,\nschematic diagram\n+ Andrews Ae$_3$ estimate\n\nNot a model input, constraint\nor validation step: runs in\nparallel with the ML branch", C_PD)
box(9.1, 0.5, 3.6, 2.3, "Streamlit interface", "User inputs composition\nand heat treatment →\npredicted properties shown\nbeside the Fe–C reference", C_APP)

arrow((3.9, 7.0), (4.6, 7.0))
arrow((6.5, 6.1), (2.1, 5.5))
arrow((3.9, 4.45), (4.6, 4.45))
# (no link between pre-processing and the experiment)
arrow((8.4, 4.45), (9.1, 4.45))
arrow((2.1, 3.4), (2.1, 2.8))
arrow((8.4, 1.65), (9.1, 1.65), ls="--")
arrow((10.9, 3.4), (10.9, 2.8))
arrow((10.9, 6.1), (10.9, 5.5))
fig.tight_layout()
fig.savefig(HERE / "figures" / "fig_workflow.png", dpi=300)
