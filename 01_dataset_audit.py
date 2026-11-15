"""Dataset audit for round-2 revision.

Answers: how many samples remain after the 0.30-0.60 wt.% C filter, how many
distinct compositions and heat-treatment routes exist, and how the data are
distributed (R1-4, R3, R4-1, unlabeled reviewer).
Row 0 of the original CSV is the in-house sample; it is separated here and
never used for training.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
FIG, RES = HERE / "figures", HERE / "results"
COMP = ["C", "Si", "Mn", "P", "S", "Ni", "Cr", "Cu", "Mo"]
PROC = ["NT", "QT", "TT"]

plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
                     "xtick.labelsize": 11, "ytick.labelsize": 11})

raw = pd.read_csv(HERE / "NIMS_Fatigue_original.csv")
inhouse = raw.iloc[[0]]
lit = raw.iloc[1:].reset_index(drop=True)

lit_f = lit[lit.C.between(0.30, 0.60)].reset_index(drop=True)
lit_f.to_csv(HERE / "literature_medium_carbon.csv", index=False)
inhouse.to_csv(HERE / "inhouse_sample.csv", index=False)

def iqr_flags(s):
    q1, q3 = s.quantile([.25, .75]); k = 1.5 * (q3 - q1)
    return ((s < q1 - k) | (s > q3 + k)).sum()

summary = {
    "rows_in_original_csv": int(len(raw)),
    "literature_rows": int(len(lit)),
    "literature_rows_after_C_filter": int(len(lit_f)),
    "removed_by_C_filter": lit[~lit.C.between(0.30, 0.60)][["C"]].to_dict("records"),
    "distinct_compositions_after_filter": int(lit_f[COMP].drop_duplicates().shape[0]),
    "distinct_heat_treatment_routes": int(lit_f[PROC].drop_duplicates().shape[0]),
    "samples_per_composition": lit_f.groupby(COMP).size().describe().round(2).to_dict(),
    "iqr_flagged_tensile": int(iqr_flags(lit_f.Tensile)),
    "iqr_flagged_hardness": int(iqr_flags(lit_f.Hardness)),
    "missing_values": int(lit_f.isna().sum().sum()),
    "ranges": lit_f[COMP + PROC + ["Tensile", "Hardness"]].describe()
              .loc[["min", "max", "mean", "std"]].round(3).to_dict(),
    "inhouse_row": inhouse.iloc[0].to_dict(),
}
(RES / "dataset_audit.json").write_text(json.dumps(summary, indent=2))

tbl = lit_f[COMP + PROC + ["Tensile", "Hardness"]].describe().T[["min", "25%", "50%", "75%", "max", "mean", "std"]]
tbl.round(3).to_csv(RES / "dataset_ranges_table.csv")

# Figure: composition and processing distributions
cols = COMP + PROC
fig, axes = plt.subplots(3, 4, figsize=(15, 10))
units = {**{c: "wt.%" for c in COMP}, **{p: "°C" for p in PROC}}
for ax, c in zip(axes.flat, cols):
    ax.hist(lit_f[c], bins=20, color="#4878a8", edgecolor="white")
    ax.axvline(inhouse[c].iloc[0], color="#c0392b", ls="--", lw=2)
    ax.set_xlabel(f"{c} ({units[c]})"); ax.set_ylabel("Count")
fig.legend(handles=[plt.Line2D([], [], color="#c0392b", ls="--", lw=2)],
           labels=["In-house sample"], loc="upper right")
fig.suptitle(f"Distribution of composition and processing variables "
             f"(literature data, n = {len(lit_f)})", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIG / "fig_dataset_distribution.png", dpi=300); plt.close(fig)

# Figure: target distributions + in-house position
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, t, u in zip(axes, ["Tensile", "Hardness"], ["MPa", "HV"]):
    ax.hist(lit_f[t], bins=25, color="#4878a8", edgecolor="white")
    ax.axvline(inhouse[t].iloc[0], color="#c0392b", ls="--", lw=2, label="In-house sample")
    ax.set_xlabel(f"{t} ({u})"); ax.set_ylabel("Count"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "fig_target_distribution.png", dpi=300); plt.close(fig)

# Figure: composition coverage, C vs Cr coloured by tensile
fig, ax = plt.subplots(figsize=(6.5, 5))
sc = ax.scatter(lit_f.C, lit_f.Cr, c=lit_f.Tensile, cmap="viridis", s=28)
ax.scatter(inhouse.C, inhouse.Cr, marker="*", s=260, c="#c0392b", label="In-house sample")
ax.set_xlabel("C (wt.%)"); ax.set_ylabel("Cr (wt.%)"); ax.legend(loc="upper left")
fig.colorbar(sc, label="Tensile strength (MPa)")
fig.tight_layout(); fig.savefig(FIG / "fig_composition_coverage_C_Cr.png", dpi=300); plt.close(fig)

print(json.dumps({k: v for k, v in summary.items() if k not in ("ranges",)}, indent=1, default=str))
