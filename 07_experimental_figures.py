"""Figures for the in-house experimental case study (details from the 2024 FYP report).

- Heat-treatment schedule: 860 °C for 1 h in a muffle furnace, then air cooling.
- Measured tensile strength and Vickers hardness before (as-rolled) and after normalising,
  each the mean of three tests (individual values were not retained, so no error bars).
"""
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
FIG = HERE / "figures"
plt.rcParams.update({"font.size": 12, "axes.labelsize": 13})

# Ae3 of the in-house composition, Andrews (1965)
comp = dict(C=0.40, Si=0.968, Mn=0.64, P=0.020, Ni=0.039, Cr=0.01, Cu=0.156, Mo=0.0, V=0.024)
ae3 = (910 - 203 * np.sqrt(comp["C"]) + 44.7 * comp["Si"] - 15.2 * comp["Ni"] + 31.5 * comp["Mo"]
       + 104 * comp["V"] - 30 * comp["Mn"] - 11 * comp["Cr"] - 20 * comp["Cu"] + 700 * comp["P"])

# --- heat-treatment schedule (schematic time axis) ---
fig, ax = plt.subplots(figsize=(8, 4.2))
t = [0, 1, 2, 2.15, 3.2]
T = [25, 860, 860, 700, 25]
ax.plot(t[:3], T[:3], color="#c0392b", lw=2.5)
tc = np.linspace(2, 3.4, 60)
ax.plot(tc, 25 + 835 * np.exp(-(tc - 2) * 3.2), color="#2471a3", lw=2.5)
ax.axhline(ae3, color="k", ls="--", lw=1.2)
ax.text(2.42, ae3 + 15, f"Ae$_3$ ≈ {ae3:.0f} °C (Andrews eq.)", fontsize=10.5)
ax.axhline(727, color="grey", ls=":", lw=1.2)
ax.text(2.42, 727 - 45, "A$_1$ = 727 °C", fontsize=10.5, color="grey")
ax.text(1.0, 300, "Heating in\nmuffle furnace", ha="center", fontsize=11)
ax.annotate("Hold at 860 °C, 1 h", (1.5, 860), xytext=(1.5, 960), ha="center", fontsize=11,
            arrowprops=dict(arrowstyle="->"))
ax.text(2.55, 380, "Air cooling to\nroom temperature", fontsize=11, color="#2471a3")
ax.set_xticks([]); ax.set_xlabel("Time (schematic)"); ax.set_ylabel("Temperature (°C)")
ax.set_ylim(0, 1050); ax.set_xlim(0, 3.4)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "fig_heat_treatment_schedule.png", dpi=300); plt.close(fig)

# --- measured properties before and after normalising ---
cond = ["As-rolled", "Normalised"]
ts, hv = [1582.49, 1471.66], [450, 260]
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, vals, lab, fmt in [(axes[0], ts, "Tensile strength (MPa)", "{:.0f}"), (axes[1], hv, "Hardness (HV)", "{:.0f}")]:
    b = ax.bar(cond, vals, color=["#7f8c8d", "#2471a3"], width=0.55)
    for rect, v in zip(b, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v * 1.01, fmt.format(v), ha="center", va="bottom", fontsize=12)
    ax.set_ylabel(lab); ax.set_ylim(0, max(vals) * 1.15)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_title("(a)", loc="left"); axes[1].set_title("(b)", loc="left")
fig.tight_layout(); fig.savefig(FIG / "fig_measured_properties.png", dpi=300); plt.close(fig)

# --- tensile specimen drawing from the submitted manuscript (dimensions confirmed by the FYP) ---
shutil.copy(HERE / "assets" / "tensile_specimen.jpg", FIG / "fig_tensile_specimen.jpg")
print(f"Ae3 = {ae3:.1f} C")
