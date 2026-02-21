# Cardiovascular Risk Prediction (10-Year CHD)

Predict **10-year Coronary Heart Disease (CHD)** risk from patient health indicators using machine learning classification.

## Goal
Build a model to predict whether a patient has a **10-year CHD risk** (`TenYearCHD`) using demographic and clinical features.

## What I did in this notebook
### 1) Data loading + exploration
- Loaded dataset from `data_cardiovascular_risk.csv`
- Performed basic EDA (shape/info/describe, duplicate check)
- Visualized feature distributions and relationship with `TenYearCHD` (example: age vs CHD)

### 2) Preprocessing
- Split features/label:
  - `X = df.iloc[:, :-1]`
  - `y = df.iloc[:, -1]` (target: `TenYearCHD`)
- Scaled numeric features using **StandardScaler** (applied column-wise)

### 3) Handling class imbalance
- Used **SMOTE** (minority oversampling) to balance the classes:
  - `SMOTE(sampling_strategy='minority', random_state=42)`
- Performed train/test split after resampling (`test_size=0.3`, `random_state=101`)

### 4) Modeling + evaluation
Trained and evaluated two main models:

**XGBoost (XGBClassifier)**
- Used **K-Fold cross-validation** (5 folds) with `cross_val_score`
- Evaluated with metrics including confusion matrix, classification report, ROC-AUC/ROC curve, precision, recall, F1

**Random Forest (RandomForestClassifier)**
- Trained baseline RF and evaluated using the same metrics
- Performed **hyperparameter tuning** using `GridSearchCV` with:
  - `n_estimators`: [100, 200, 300]
  - `max_depth`: [None, 4, 8, 10]
  - `min_samples_split`: [2, 3, 5, 10]
  - `min_samples_leaf`: [2, 3, 4]
  - `max_features`: ['auto', 'sqrt', 'log2']
- Printed best parameters from GridSearchCV

## Tech stack
- Python, Jupyter Notebook
- Pandas, NumPy
- scikit-learn (StandardScaler, metrics, RandomForest, GridSearchCV, KFold)
- imbalanced-learn (SMOTE)
- XGBoost (XGBClassifier)
- Matplotlib / Seaborn (plots)

## Files
- `cardiovascular_risk_prediction.ipynb` — main notebook
- `data_cardiovascular_risk.csv` — dataset (place in the same folder or update the path in the notebook)

## How to run
```bash
python -m venv .venv
# mac/linux
source .venv/bin/activate
# windows
# .venv\Scripts\activate

pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn jupyter

jupyter notebook
