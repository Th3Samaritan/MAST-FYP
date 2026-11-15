# -*- coding: utf-8 -*-
"""Core logic of the decision-support tool (no Streamlit dependency, so it can be tested directly).

- predict(): tensile strength (MPa) and hardness (HV) from the two final XGBoost models
- out_of_range(): inputs outside the training ranges (predictions there are extrapolations)
- andrews_ae3(): Ae3 estimate for the actual composition (Andrews 1965)
- fe_c_diagram(): schematic metastable Fe–Fe3C diagram (binary, equilibrium; standard tabulated points)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

HERE = Path(__file__).parent
FEATURES = ["C", "Si", "Mn", "P", "S", "Ni", "Cr", "Cu", "Mo", "NT", "QT", "TT"]
UNITS = {**{k: "wt.%" for k in FEATURES[:9]}, "NT": "°C", "QT": "°C", "TT": "°C"}
RANGES = json.loads((HERE / "training_ranges.json").read_text())


def load_models():
    models = {}
    for target in ("Tensile", "Hardness"):
        m = XGBRegressor()
        m.load_model(str(HERE / "models" / f"xgb_final_{target}.json"))
        models[target] = m
    return models


def predict(models, inputs):
    x = pd.DataFrame([[float(inputs[f]) for f in FEATURES]], columns=FEATURES)
    return {"Tensile strength (MPa)": float(models["Tensile"].predict(x)[0]),
            "Hardness (HV)": float(models["Hardness"].predict(x)[0])}


def out_of_range(inputs):
    return [(f, inputs[f], RANGES[f]["min"], RANGES[f]["max"]) for f in FEATURES
            if not RANGES[f]["min"] <= inputs[f] <= RANGES[f]["max"]]


def andrews_ae3(c):
    """Andrews (1965) Ae3 in °C; elements absent from the dict are taken as zero."""
    g = lambda k: float(c.get(k, 0.0))
    return (910 - 203 * np.sqrt(g("C")) + 44.7 * g("Si") - 15.2 * g("Ni") + 31.5 * g("Mo") + 104 * g("V")
            - 30 * g("Mn") - 11 * g("Cr") - 20 * g("Cu") + 700 * g("P"))


# Standard invariant points of the metastable Fe–Fe3C system (wt.% C, °C)
_S, _P, _E = (0.76, 727), (0.022, 727), (2.14, 1147)
_H, _J, _B, _A = (0.09, 1495), (0.17, 1495), (0.53, 1495), (0.0, 1538)
_XMAX = 2.2
_a3 = PchipInterpolator([0.0, 0.2, 0.4, 0.6, 0.76], [912, 851, 800, 757, 727])
_liq = lambda c: 1495 + (1147 - 1495) * (c - 0.53) / (4.30 - 0.53)


def binary_a3(c):
    return float(_a3(min(c, 0.76)))


def fe_c_diagram(comp, NT=None, QT=None, TT=None, label=None, figsize=(10, 8.3)):
    C0 = float(comp["C"])
    c_a3 = np.linspace(0, 0.76, 80)
    fig, ax = plt.subplots(figsize=figsize)
    fill = dict(alpha=0.18, lw=0)
    ax.add_patch(Polygon([(0, 912), *zip(c_a3, _a3(c_a3)), _S, _E, _J, (0, 1394)], color="#2e86c1", **fill))
    ax.add_patch(Polygon([(0, 912), *zip(c_a3, _a3(c_a3)), _P, (0, 727)], color="#27ae60", **fill))
    ax.add_patch(Polygon([_S, _E, (_XMAX, 1147), (_XMAX, 727)], color="#8e44ad", **fill))
    ax.add_patch(Polygon([_P, (_XMAX, 727), (_XMAX, 400), (0.005, 400)], color="#7f8c8d", **fill))
    ax.add_patch(Polygon([_J, _B, (_XMAX, _liq(_XMAX)), (_XMAX, 1147), _E], color="#e67e22", **fill))
    ax.add_patch(Polygon([_B, _A, (0, 1600), (_XMAX, 1600), (_XMAX, _liq(_XMAX))], color="#f1c40f", **fill))

    k = dict(color="k", lw=2)
    ax.plot(c_a3, _a3(c_a3), **k)
    ax.plot([_S[0], _E[0]], [_S[1], _E[1]], **k)
    ax.plot([_P[0], _XMAX], [727, 727], **k)
    ax.plot([_E[0], _XMAX], [1147, 1147], **k)
    ax.plot([_J[0], _E[0]], [_J[1], _E[1]], **k)
    ax.plot([_A[0], _B[0], _XMAX], [_A[1], _B[1], _liq(_XMAX)], **k)
    ax.plot([_H[0], _B[0]], [1495, 1495], **k)
    for xs, ys in [([0, _H[0]], [1394, 1495]), ([0, _J[0]], [1394, 1495]), ([0, _P[0]], [912, 727]),
                   ([0.022, 0.015, 0.008, 0.005], [727, 600, 500, 400])]:
        ax.plot(xs, ys, color="k", lw=1.2)

    lab = dict(fontsize=13, ha="center", va="center")
    ax.text(1.05, 1000, "γ (austenite)", **lab)
    ax.text(0.22, 760, "α + γ", **lab)
    ax.text(1.55, 900, "γ + Fe$_3$C", **lab)
    ax.text(1.30, 500, "α + Fe$_3$C\n(pearlite and proeutectoid\nconstituents on slow cooling)",
            fontsize=11, ha="center")
    ax.text(1.55, 1330, "L + γ", **lab)
    ax.text(1.00, 1555, "L (liquid)", **lab)
    ax.text(0.07, 1445, "δ + γ", fontsize=9, ha="center")
    ax.text(0.04, 600, "α", **lab)
    ax.plot(*_S, "o", color="#c0392b", ms=9, zorder=5)
    ax.annotate("Eutectoid point\n0.76 wt.% C, 727 °C", _S, xytext=(0.95, 760), fontsize=11,
                arrowprops=dict(arrowstyle="->", color="#c0392b"), color="#c0392b")
    ax.text(_XMAX - 0.02, 735, "A$_1$", ha="right", fontsize=12)
    ax.text(0.47, 830, "A$_3$", fontsize=12)
    ax.text(1.35, 1000, "A$_{cm}$", fontsize=12, rotation=40)

    ae3 = andrews_ae3(comp)
    ax.axvline(C0, color="#c0392b", ls="--", lw=1.8, label=label or f"Selected steel, {C0:.2f} wt.% C")
    ax.plot(C0, binary_a3(C0), "s", color="k", ms=7, label=f"Binary A$_3$ at {C0:.2f} wt.% C ≈ {binary_a3(C0):.0f} °C")
    ax.plot(C0, ae3, "D", color="#2471a3", ms=7, label=f"Ae$_3$ (Andrews, this composition) ≈ {ae3:.0f} °C")
    if NT is not None:
        ax.plot(C0, NT, "*", color="#d35400", ms=16, label=f"Normalising, {NT:.0f} °C")
    if QT is not None:
        ax.plot(C0, QT, "v", color="#7d3c98", ms=10, label=f"Austenitising before quench, {QT:.0f} °C")
    if TT is not None:
        ax.plot(C0, TT, "^", color="#117a65", ms=10,
                label=f"Tempering, {TT:.0f} °C" + (" (below A$_1$)" if TT < 727 else ""))
    ax.set_xlim(0, _XMAX); ax.set_ylim(400, 1600)
    ax.set_xlabel("Carbon (wt.%)"); ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=10.5, frameon=False)
    ax.set_title("Metastable Fe–Fe$_3$C diagram (binary, equilibrium; steel region)", fontsize=14)
    fig.tight_layout()
    return fig, ae3
