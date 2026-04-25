AI-Driven Screening for Carbon Capture Materials
Predicting CO₂ uptake in Metal-Organic Frameworks (MOFs) using Machine Learning.

Problem Statement
Over 90,000+ MOF structures exist, but experimentally testing each one for CO₂ adsorption is slow and expensive. This project trains ML models to predict CO₂ uptake purely from geometric structural properties — enabling rapid screening of thousands of candidates without lab experiments.

Dataset

Source: Computational MOF Screening Database
Size: 324,426 MOF structures
Target: CO₂ uptake at flue gas conditions (0.15 bar, 298 K) in mmol/g
Input features: Surface area, void fraction, void volume, pore size, volume, weight, topology, functional groups, metal linker, organic linkers


Pipeline
Data Loading
    ↓
EDA — distributions, correlations, missing values
    ↓
Cleaning — drop error columns, leakage columns, nulls
    ↓
Encoding — frequency encoding, one-hot encoding
    ↓
Feature Engineering — interaction features
    ↓
Train / Val / Test Split — 70% / 15% / 15%
    ↓
StandardScaler — fit on train only
    ↓
log1p transform on target
    ↓
Model Training — LightGBM, XGBoost, Random Forest, FFN
    ↓
Evaluation — R², MSE, MAE
    ↓
Scatter plots, Residuals, Feature Importance

Models and Results
ModelTrain R²Test R²Test MSETest MAELightGBM0.95420.92660.02110.0813FFN / MLP0.93400.91600.0230—XGBoost0.91350.9095—0.1485Random Forest0.94810.90420.02700.0940
Best model: LightGBM with Test R² = 0.9266

Key Design Decisions
Why drop leakage columns?
Columns like working capacity and CO₂/N₂ selectivity are simulation outputs themselves. Keeping them gives a fake high R² — the model learns shortcuts instead of real structure-property relationships.
Why log1p transform on target?
CO₂ uptake is right-skewed. Log transforming compresses the scale and helps the model fit all MOFs equally well instead of chasing high-uptake outliers. Predictions are reversed with expm1() before evaluation.
Why StandardScaler?
Features like surface area (thousands of m²/g) and void fraction (0 to 1) are on very different scales. Scaling is critical for FFN and useful for pipeline consistency across all models.

Tech Stack

Python, Pandas, NumPy
Matplotlib, Seaborn
Scikit-learn
LightGBM
XGBoost
PyTorch / TensorFlow (FFN)


How to Run
bashgit clone https://github.com/aashima2310/carbon_capture_project.git
cd carbon_capture_project
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost
jupyter notebook carbonproject.ipynb
Place all_MOFs_screening_data.csv in the same folder before running.

Repository Structure
carbon_capture_project/
├── carbonproject.ipynb
├── all_MOFs_screening_data.csv
└── README.md

Future Work

Hyperparameter tuning with Optuna
SHAP values for feature interpretability
Stacking ensemble across all 4 models
Graph Neural Networks on crystal structure graphs
