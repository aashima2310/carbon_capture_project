#  AI-Driven Screening for Carbon Capture Materials

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Random%20Forest-228B22?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/FFN%20%2F%20MLP-8B008B?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
</p>

<p align="center">
  <b>Domain:</b> Chemical Engineering &nbsp;|&nbsp; <b>Theme:</b> Materials Discovery &nbsp;|&nbsp; <b>Task:</b> Regression
</p>

---

##  Overview

**Metal-Organic Frameworks (MOFs)** are nanoporous crystalline materials with extraordinarily high internal surface areas — a single gram can exceed the surface area of a football field. These structural properties make them promising candidates for selectively adsorbing CO₂ from industrial flue gas streams, a critical step in **Carbon Capture and Storage (CCS)** technologies.

Over **90,000+ MOF structures** have been synthesized to date, yet experimentally measuring the CO₂ uptake capacity of each candidate remains prohibitively slow and resource-intensive.

> **Can a data-driven model learn the relationship between a MOF's structural geometry and its CO₂ uptake capacity — and use that knowledge to rapidly screen thousands of untested candidates?**

This project answers that question by training and comparing **four machine learning models** that predict CO₂ uptake purely from geometric structural descriptors.

---

##  Dataset

| Property | Details |
|---|---|
| Source | Computational MOF Screening Database |
| Total Structures | 324,426 MOFs |
| Total Features | 42 columns |
| Target Variable | `CO2_uptake_P0.15bar_T298K [mmol/g]` |
| Conditions | Flue gas — 0.15 bar, 298 K |

### Input Features Used

| Feature | Description |
|---|---|
| `surface_area [m²/g]` | Internal surface area of the MOF |
| `void_fraction` | Fraction of empty pore space |
| `void_volume [cm³/g]` | Total pore volume |
| `volume [Å³]` | Unit cell volume |
| `weight [u]` | Molecular weight |
| `largest_free_sphere_diameter [Å]` | Largest sphere that can pass through pores |
| `largest_included_sphere_diameter [Å]` | Largest sphere that fits inside pores |
| `topology` | Framework connectivity type (11 unique) |
| `functional_groups` | Chemical functional groups (~400 unique) |
| `metal_linker` | Metal node type (7 unique) |
| `organic_linker1/2` | Organic linker types |

---

##  Methodology

### Data Cleaning
- Dropped all **error/uncertainty columns** — measure simulation convergence, not MOF properties
- Dropped all **leakage columns** — simulation outputs at other conditions (working capacity, selectivity, uptake at 0.10/0.70 bar) that would give artificially high R²
- Dropped `MOFname` identifier
- Removed ~2000 null rows (<1% of data)

### Encoding Strategy

| Column | Strategy | Reason |
|---|---|---|
| `functional_groups` | Frequency Encoding | ~400 unique values — too many for one-hot |
| `topology` | One-Hot Encoding | 11 values, no natural order |
| `metal_linker` | One-Hot Encoding | 7 values, no natural order |
| `organic_linker1/2` | Already integers | No encoding needed |

### Feature Engineering
Three interaction features were created from geometric descriptors:
- `sa_per_volume` — surface area packed per unit volume
- `pore_density` — void fraction relative to pore size
- `void_x_sa` — combined adsorption capacity proxy

### Train / Validation / Test Split

| Split | Percentage | Rows |
|---|---|---|
| Train | 70% | ~227,000 |
| Validation | 15% | ~48,600 |
| Test | 15% | ~48,600 |

### Why `log1p` Transform on Target?
CO₂ uptake is **right-skewed** — most MOFs have low uptake but a few have very high values. Training on raw skewed targets causes the model to chase outliers and fit the bulk of data poorly. Applying `log1p(y)` compresses the scale and makes the distribution symmetric, directly improving R². Predictions are reversed with `expm1()` before evaluation.

### Why `StandardScaler`?
Features span vastly different ranges — `surface_area` can be in the thousands while `void_fraction` is between 0 and 1. StandardScaler normalizes all features to mean=0, std=1. **Critical for FFN** and applied consistently across all models for fair comparison.

---

##  Models

###  LightGBM
Gradient boosting with **leaf-wise tree growth** and histogram binning. Extremely fast on large datasets. Uses early stopping on the validation set to prevent overfitting.

###  XGBoost
Gradient boosting with **level-wise tree growth** and built-in L1/L2 regularization. Robust generalization — smallest train/test gap of all models.

###  Random Forest
Bagging ensemble of decision trees. Each tree is trained on a random subset of data and features — reduces variance through averaging.

###  FFN / MLP
Multi-layer perceptron trained with Adam optimizer. StandardScaler is critical here — unlike tree models, neural networks require scaled inputs for stable gradient descent training.

---

## 📊 Results

| Model | Train R² | Test R² | Test MSE | Test MAE |
|---|---|---|---|---|
| 🥇 **LightGBM** | **0.9542** | **0.9266** | **0.0211** | **0.0813** |
| 🥈 **FFN / MLP** | 0.9340 | 0.9160 | 0.0230 | — |
| 🥉 **XGBoost** | 0.9135 | 0.9095 | — | 0.1485 |
| **Random Forest** | 0.9481 | 0.9042 | 0.0270 | 0.0940 |

### Key Observations
- **LightGBM** achieves the highest test R² (0.9266) and lowest MSE — best overall performer
- **XGBoost** shows the smallest train-test gap (0.004) — best generalization across all models
- **FFN** performs competitively at 0.916 without manual tree-based tuning
- **Random Forest** has the largest test MSE — ensemble averaging loses accuracy on high-uptake MOFs

---

## 🛠️ Tech Stack
Python 3.10     pandas      numpy       matplotlib
seaborn         scikit-learn            lightgbm
xgboost         PyTorch / TensorFlow    Jupyter Notebook

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/aashima2310/carbon_capture_project.git
cd carbon_capture_project

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost

# 3. Launch notebook
jupyter notebook carbonproject.ipynb
```

> ⚠️ Place `all_MOFs_screening_data.csv` in the same directory as the notebook before running.

---

---

## 🔮 Future Work

- Hyperparameter tuning with **Optuna** for all 4 models
- **Stacking ensemble** — LightGBM + XGBoost + FFN
- **Graph Neural Networks** using MOF crystal structure graphs
- Include Henry's constant and isosteric heat as additional descriptors

---
