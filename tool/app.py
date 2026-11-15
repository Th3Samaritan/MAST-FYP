# -*- coding: utf-8 -*-
"""Streamlit decision-support tool described in the manuscript (Section 2.6).

Run:  streamlit run app.py
"""
import streamlit as st

from tool_core import FEATURES, RANGES, UNITS, andrews_ae3, binary_a3, fe_c_diagram, load_models, out_of_range, predict

st.set_page_config(page_title="Medium-carbon steel property predictor", layout="wide")
st.title("Tensile strength and hardness of quenched-and-tempered medium-carbon steels")
st.caption("XGBoost models trained on 357 NIMS records (0.30–0.60 wt.% C), each normalised, quenched and tempered. "
           "Expected error for a new composition within the training ranges: about 26 MPa and 8 HV "
           "(mean absolute error, composition-grouped cross-validation). The models apply only to normalised, "
           "quenched-and-tempered steels; they are not valid for steels that are only normalised or as-rolled.")

models = st.cache_resource(load_models)()
DEFAULT = {"C": 0.40, "Si": 0.26, "Mn": 0.77, "P": 0.018, "S": 0.015, "Ni": 0.05, "Cr": 0.98, "Cu": 0.13,
           "Mo": 0.17, "NT": 870.0, "QT": 855.0, "TT": 550.0}

with st.sidebar:
    st.header("Composition (wt.%)")
    inputs = {}
    for f in FEATURES[:9]:
        inputs[f] = st.number_input(f, min_value=0.0, max_value=5.0, value=DEFAULT[f], step=0.01, format="%.3f",
                                    help=f"Training range {RANGES[f]['min']}–{RANGES[f]['max']} {UNITS[f]}")
    st.header("Heat treatment (°C)")
    names = {"NT": "Normalising temperature", "QT": "Austenitising temperature before quench",
             "TT": "Tempering temperature"}
    for f in ["NT", "QT", "TT"]:
        inputs[f] = st.number_input(names[f], min_value=300.0, max_value=1100.0, value=DEFAULT[f], step=5.0,
                                    help=f"Training range {RANGES[f]['min']:.0f}–{RANGES[f]['max']:.0f} °C")

pred = predict(models, inputs)
oor = out_of_range(inputs)

col1, col2 = st.columns([1, 1.6])
with col1:
    st.subheader("Predicted properties")
    for k, v in pred.items():
        st.metric(k, f"{v:.0f}")
    if oor:
        st.warning("Outside the training range, so these predictions are extrapolations and may be unreliable:\n\n"
                   + "\n".join(f"- {f} = {v:g} {UNITS[f]} (training range {lo:g}–{hi:g})" for f, v, lo, hi in oor))
    else:
        st.success("All inputs are within the training ranges.")
    st.subheader("Transformation temperatures")
    st.write(f"Binary A₃ at {inputs['C']:.2f} wt.% C: **{binary_a3(inputs['C']):.0f} °C**")
    st.write(f"Ae₃ for this composition (Andrews 1965): **{andrews_ae3(inputs):.0f} °C**")
    st.caption("The diagram is a binary, equilibrium reference. It is not used by the prediction models and does "
               "not show martensite or tempered microstructures, which govern the predicted properties.")
with col2:
    fig, _ = fe_c_diagram(inputs, inputs["NT"], inputs["QT"], inputs["TT"])
    st.pyplot(fig)
