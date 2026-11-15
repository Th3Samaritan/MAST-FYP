import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# Load the trained model
model = XGBRegressor()
model.load_model('xgb_model.json')  # Load the model saved in JSON format

# Constants for heat treatment temperatures
NORMALIZING_TEMP = 865  # Degrees Celsius
QUENCHING_TEMP = 865  # Degrees Celsius
TEMPERING_TEMP = 550  # Degrees Celsius

# Constants for phase diagram boundaries
solidus_curve = [
    (0, 1538), (0.76, 723), (2.06, 1130), (6.67, 1130)
]  # Solidus line points (Carbon %, Temperature °C)
liquidus_curve = [
    (0, 1538), (0.16, 1495), (0.76, 1495), (2.06, 1270), (6.67, 1147)
]  # Liquidus line points (Carbon %, Temperature °C)
eutectic_point = (0.76, 723)  # Eutectic point at 0.76% carbon and 723°C

# Function to generate the iron-carbon phase diagram with fixed limits and annotations
def generate_phase_diagram(carbon_composition):
    solidus_x, solidus_y = zip(*solidus_curve)
    liquidus_x, liquidus_y = zip(*liquidus_curve)
    liquidus_x_trunc = liquidus_x[:4]
    liquidus_y_trunc = liquidus_y[:4]

    plt.figure(figsize=(8, 6))

    plt.plot(solidus_x, solidus_y, label="Solidus Line", color='blue')
    plt.plot(liquidus_x_trunc, liquidus_y_trunc, label="Liquidus Line", color='orange')
    plt.scatter(*eutectic_point, color='red', zorder=5)
    plt.text(0.8, 700, "Eutectic Point", color='red')

    plt.plot([carbon_composition, carbon_composition], [0, 1600], 'r--', label=f'C={carbon_composition:.4f}%')

    plt.fill_betweenx(solidus_y, solidus_x, liquidus_x_trunc, where=(np.array(solidus_x) <= np.array(liquidus_x_trunc)),
                      interpolate=True, color='lightgray', alpha=0.5, label="Solid + Liquid Region")

    plt.fill_betweenx([0, 723], 0, 0.02, color='lightgreen', alpha=0.3, label="Ferrite (α)")
    plt.fill_betweenx([0, 723], 0.02, 0.76, color='cyan', alpha=0.3, label="Pearlite")
    plt.fill_betweenx([723, 900], 0, 0.76, color='green', alpha=0.3, label="Austenite (γ)")
    plt.fill_betweenx([723, 1495], 0.76, 2.1, color='purple', alpha=0.3, label="Austenite + Cementite (γ+Fe3C)")
    plt.fill_betweenx([1495, 1600], 0, 6.67, color='yellow', alpha=0.3, label="Liquid")
    plt.fill_betweenx([500, 900], 0, 2.06, color='lightblue', alpha=0.3, label="Bainite")

    plt.text(0.1, 400, "Ferrite (α): Soft, ductile phase, low carbon content.", fontsize=10, color='darkgreen')
    plt.text(0.2, 600, "Pearlite: Mixture of ferrite and cementite, stronger but less ductile.", fontsize=10, color='darkcyan')
    plt.text(0.5, 1000, "Austenite (γ): High-temperature phase, allows carbon to dissolve.", fontsize=10, color='darkgreen')
    plt.text(1.5, 1100, "Bainite: Tough and strong, forms at lower temperatures.", fontsize=10, color='darkblue')
    plt.text(4, 1400, "Liquid: Fully molten state of iron and carbon.", fontsize=10, color='darkgoldenrod')

    plt.xlim(0.0, 6.67)
    plt.ylim(0, 1500)

    plt.title("Iron-Carbon Phase Diagram with Detailed Phases")
    plt.xlabel("Carbon Composition (%)")
    plt.ylabel("Temperature (°C)")
    plt.axvline(x=carbon_composition, color='red', linestyle='--')
    plt.text(carbon_composition + 0.05, 1000, f"C={carbon_composition:.4f}%", color='red')

    plt.legend(loc="upper right")
    plt.grid(True)

    st.pyplot(plt)

# Streamlit interface
st.title("Steel Heat Treatment Prediction and Iron-Carbon Phase Diagram")

# User input for elemental composition
st.header("Elemental Composition Input")
C = st.number_input('Carbon (C)', min_value=0.0, max_value=2.0, value=0.40, step=0.0001, format="%.2f")
Si = st.number_input('Silicon (Si)', min_value=0.0, max_value=2.0, value=0.968, step=0.0001, format="%.3f")
Mn = st.number_input('Manganese (Mn)', min_value=0.0, max_value=2.0, value=0.64, step=0.0001, format="%.3f")
P = st.number_input('Phosphorus (P)', min_value=0.0, max_value=0.5, value=0.02, step=0.0001, format="%.3f")
S = st.number_input('Sulfur (S)', min_value=0.0, max_value=0.5, value=0.129, step=0.0001, format="%.3f")
Ni = st.number_input('Nickel (Ni)', min_value=0.0, max_value=2.0, value=0.039, step=0.0001, format="%.3f")
Cr = st.number_input('Chromium (Cr)', min_value=0.0, max_value=2.0, value=0.01, step=0.0001, format="%.3f")
Cu = st.number_input('Copper (Cu)', min_value=0.0, max_value=2.0, value=0.156, step=0.0001, format="%.3f")
Mo = st.number_input('Molybdenum (Mo)', min_value=0.0, max_value=2.0, value=0.0, step=0.0001, format="%.2f")

# User input for heat treatment method
st.header("Heat Treatment Method")
heat_treatment = st.selectbox("Select the Heat Treatment Method", ("Normalizing", "Quenching", "Tempering"))

# Assign temperature based on the selected heat treatment method
if heat_treatment == "Normalizing":
    temperature = st.number_input('Normalizing Temperature (°C)', value=NORMALIZING_TEMP, min_value=800, max_value=900, step=1)
elif heat_treatment == "Quenching":
    temperature = st.number_input('Quenching Temperature (°C)', value=QUENCHING_TEMP, min_value=800, max_value=900, step=1)
elif heat_treatment == "Tempering":
    temperature = st.number_input('Tempering Temperature (°C)', value=TEMPERING_TEMP, min_value=500, max_value=600, step=1)

# Prediction Button
if st.button("Predict"):
    new_data = {
        'C': C, 'Si': Si, 'Mn': Mn, 'P': P, 'S': S,
        'Ni': Ni, 'Cr': Cr, 'Cu': Cu, 'Mo': Mo,
        'NT': 1 if heat_treatment == "Normalizing" else 0,
        'TT': 1 if heat_treatment == "Tempering" else 0,
        'QT': 1 if heat_treatment == "Quenching" else 0
    }
    input_df = pd.DataFrame(new_data, index=[0])

    predictions = model.predict(input_df)

    tensile_strength = float(predictions[0][0])
    hardness = float(predictions[0][1])
    fatigue_strength = float(predictions[0][2])

    st.subheader("Model Predictions")
    st.write(f"Tensile Strength: {tensile_strength:.2f}")
    st.write(f"Hardness: {hardness:.2f}")
    st.write(f"Fatigue Strength: {fatigue_strength:.2f}")

    st.subheader("Iron-Carbon Phase Diagram")
    generate_phase_diagram(C)
