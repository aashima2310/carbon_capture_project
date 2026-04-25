🧪 AI-Driven Screening for Carbon Capture Materials
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LightGBM-✓-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/XGBoost-✓-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Random_Forest-✓-teal?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/FFN / MLP-✓-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Domain-Chemical_Engineering-red?style=for-the-badge"/>
</p>

📌 Project Overview
Metal-Organic Frameworks (MOFs) are nanoporous crystalline materials with extraordinarily high internal surface areas — a single gram can exceed the surface area of a football field. These structural properties make them promising candidates for selectively adsorbing CO₂ from industrial flue gas streams, a critical step in Carbon Capture and Storage (CCS) technologies.
Over 90,000+ MOF structures have been synthesized to date, yet experimentally measuring the CO₂ uptake capacity of each candidate remains prohibitively slow and resource-intensive.

Central Question: Can a data-driven model learn the relationship between a MOF's structural geometry and its CO₂ uptake capacity — and use that knowledge to rapidly screen thousands of untested candidates?

This project builds and compares four machine learning models that predict CO₂ uptake purely from geometric structural descriptors, enabling rapid in-silico screening without expensive simulations.

📂 Dataset
PropertyValueSourceComputational MOF Screening DatabaseTotal MOFs324,426 structuresFeaturesGeometric descriptors (surface area, void fraction, pore size, volume, weight)TargetCO2_uptake_P0.15bar_T298K [mmol/g] — CO₂ uptake at flue gas conditionsCategorical featuresTopology, functional groups, metal linker, organic linkers
Key Features Used

surface_area [m²/g] — internal surface area of the MOF
void_fraction — fraction of empty pore space
void_volume [cm³/g] — total pore volume
volume [Å³] — unit cell volume
weight [u] — molecular weight
largest_free_sphere_diameter [Å] — largest sphere that can pass through pores
largest_included_sphere_diameter [Å] — largest sphere that fits inside pores
topology, functional_groups, metal_linker, organic_linker1/2 — structural descriptors


🔬 Methodology
Pipeline
Raw Data (324,426 MOFs)
        │
        ▼
  EDA & Visualization
        │
        ▼
  Data Cleaning
  ├── Drop error/uncertainty columns
  ├── Drop leakage columns (simulation outputs at other conditions)
  ├── Drop MOFname identifier
  └── Drop null rows (<1% of data)
        │
        ▼
  Feature Engineering
  ├── Frequency encoding  → functional_groups (~400 unique)
  ├── One-hot encoding    → topology (11), metal_linker (7)
  └── Interaction features → sa_per_volume, pore_density, void_x_sa
        │
        ▼
  Train / Val / Test Split  →  70% / 15% / 15%
        │
        ▼
  StandardScaler (fit on train only)
        │
        ▼
  log1p(y) Target Transform
        │
        ▼
  Model Training × 4
  ├── LightGBM
  ├── XGBoost
  ├── Random Forest
  └── FFN / MLP
        │
        ▼
  Evaluation → R², MSE, MAE
        │
        ▼
  Diagnostic Plots → Scatter, Residuals, Feature Importance
Why log1p Transform on Target?
CO₂ uptake is right-skewed — most MOFs have low uptake but a few have very high values. Training on the raw target lets the model chase outliers. Applying log1p(y) compresses the scale, makes the distribution symmetric, and directly improves R². Predictions are reversed with expm1() before evaluation.
Why No Leakage Columns?
Columns like working_capacity, CO2/N2_selectivity, and uptake at other conditions are themselves simulation outputs. Keeping them creates data leakage — the model learns shortcuts instead of structure-property relationships, producing artificially high R² that collapses on truly unseen MOFs.

🤖 Models
1. 🌲 LightGBM
Gradient boosting with leaf-wise tree growth and histogram binning. Extremely fast on large datasets. Uses early stopping on validation set.
2. ⚡ XGBoost
Gradient boosting with level-wise tree growth and built-in regularization. Robust and well-calibrated across train/test.
3. 🌳 Random Forest
Bagging ensemble of decision trees. Each tree trained on a random subset of data and features. Resistant to overfitting through averaging.
4. 🧠 FFN / MLP (Feed-Forward Neural Network)
Multi-layer perceptron trained with Adam optimizer. StandardScaler is critical here — unlike tree models, neural networks require scaled inputs for stable gradient descent.

📊 Results
ModelTrain R²Test R²Test MSETest MAE🌲 LightGBM0.95420.92660.02110.0813🧠 FFN / MLP0.93400.91600.0230—⚡ XGBoost0.91350.9095—0.1485🌳 Random Forest0.94810.90420.02700.0940

Best Model: LightGBM with Test R² = 0.9266

Key Observations

LightGBM achieves the highest test R² and lowest MSE — best overall performer
XGBoost shows the smallest train-test gap (0.004) — best generalization
Random Forest has the largest test MSE — ensemble averaging smooths predictions but loses accuracy on high-uptake MOFs
FFN performs competitively without any manual feature engineering tuning


📈 Visualizations
The notebook includes:

Target distribution (raw vs log-transformed)
Geometric feature histograms
Correlation heatmap
CO₂ uptake by topology (boxplot)
Predicted vs Actual scatter plots (all 4 models)
Residuals plots
Feature importance (LightGBM & XGBoost)


🛠️ Tech Stack
LibraryPurposepandasData loading and manipulationnumpyNumerical operationsmatplotlib / seabornVisualizationscikit-learnPreprocessing, splitting, metricslightgbmLightGBM modelxgboostXGBoost modeltorch / tensorflowFFN / MLP model

🚀 How to Run
bash# Clone the repository
git clone https://github.com/aashima2310/carbon_capture_project.git
cd carbon_capture_project

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost

# Open the notebook
jupyter notebook carbonproject.ipynb



🔮 Future Improvements

Optuna hyperparameter search for all models
SHAP values for deeper feature interpretability
Stacking ensemble (LightGBM + XGBoost + FFN)
Graph Neural Networks using MOF crystal structure graphs
Include Henry's constant and isosteric heat as additional structural descriptors
