# Tensile strength and hardness of medium-carbon steels: code, data and tool

This repository holds the code, data and decision-support tool for the manuscript *Predictive Simulation of Mechanical Properties and Iron-Carbon Phase Diagram Region for Steel Using Materials Informatics* (Discover Applied Sciences, under review). It reproduces every number, table and figure in the revised manuscript.

## Data

| File | Contents |
|---|---|
| `NIMS_Fatigue_original.csv` | The 361-record file used in the project. Rows 1–360 are the NIMS fatigue data set (normalised, quenched and tempered carbon and low-alloy steels; see Xiong, Zhang & Shi, *Sci. China Tech. Sci.* 63, 1247, 2020). Row 0 is the in-house steel. |
| `literature_medium_carbon.csv` | The 357 NIMS records with 0.30–0.60 wt.% C used for training and validation (written by `01_dataset_audit.py`). |
| `inhouse_sample.csv` | The in-house 0.40 wt.% C steel (normalised at 860 °C for 1 h, air cooled; mean of three tests). It is **never used for training or tuning**, only as an external test. |

Units: composition in wt.%, temperatures in °C, tensile strength in MPa, hardness in Vickers (HV).

## Reproducing the analysis

```
pip install -r requirements.txt
python 01_dataset_audit.py
python 02_models.py            # about 3 minutes
python 03_interpretation.py
python 04_phase_diagram.py
python 05_workflow_figure.py
python 06_composite_figures.py
python 07_experimental_figures.py
```

| Script | Output |
|---|---|
| `01_dataset_audit.py` | Medium-carbon subset, variable ranges (`results/dataset_ranges_table.csv`), distribution figures |
| `02_models.py` | Random forest, ANN and XGBoost per target under three validation protocols: random 80/20 split, nested random 5-fold CV and nested composition-grouped 5-fold CV (`results/table1_performance_formatted.csv`); selected hyperparameters; final XGBoost models; prediction of the in-house steel (`results/inhouse_holdout_predictions.csv`) |
| `03_interpretation.py` | Gain importance, SHAP summary and dependence plots, rank table, parity and residual plots |
| `04_phase_diagram.py` | Metastable Fe–Fe₃C reference diagram with the Andrews (1965) Ae₃ for a given composition |
| `05_workflow_figure.py` | Workflow figure |
| `06_composite_figures.py` | Multi-panel SHAP figures |
| `07_experimental_figures.py` | Heat-treatment schedule, measured properties and tensile specimen figures |

Every composition in the NIMS data appears at three tempering temperatures, so random cross-validation places near-duplicates of each test steel in the training folds. The composition-grouped protocol keeps all records of a composition in the same fold. Its results are the ones to use as the expected error for a new steel.

Tested with Python 3.12, scikit-learn 1.8.0, XGBoost 3.2.0 and SHAP 0.52.0. `figures/` and `results/` hold the outputs as used in the manuscript.

## Decision-support tool

```
cd tool
pip install -r requirements.txt
streamlit run app.py
```

The tool predicts tensile strength and hardness from composition and heat-treatment temperatures with the final XGBoost models (`tool/models/`). It flags any input outside the training ranges (`tool/training_ranges.json`) and draws the Fe–Fe₃C reference diagram with the chosen temperatures. The models apply only to normalised, quenched-and-tempered steels inside these ranges.

## Earlier version

`legacy_2024/` keeps the original 2024 final-year-project code (`FYP finale.ipynb`, `finalee.py`) for reference. It trained one multi-output model on a random split that included the in-house steel. It is superseded by the scripts above and is not the analysis reported in the revised manuscript.
