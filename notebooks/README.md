
### 2. notebooks/README.md (inside notebooks folder)

```markdown
# Notebooks Folder README

This folder contains all Jupyter notebooks for the project.

### Available Notebooks
- **eda-fraud-data.ipynb**  
  Complete EDA for e-commerce data (Fraud_Data.csv) – geolocation, time patterns, purchase behavior

- **eda-creditcard.ipynb**  
  EDA for anonymized bank data (creditcard.csv) – Amount, Time, V-features distributions

- **feature-engineering.ipynb**  
  Task 1b: Full pipeline – geolocation, velocity/time features, scaling + encoding, SMOTE (50/50 train), imbalance table saved

- **modeling.ipynb**  
  Task 2: Baseline Logistic Regression + Random Forest champion – PR-AUC 0.8990, F1 0.7021, **12 false positives**, CV, tuning

- **shap_explainability.ipynb**  
  Task 3: SHAP analysis – built-in importance, summary plot, force plots, top 5 drivers, 3 business recommendations

### How to Use
1. Launch Jupyter:
   ```bash
   jupyter notebook
Run in order: EDA → feature-engineering → modeling → shap_explainability
Select "Python (venv)" kernel
All figures saved to reports/figures/.
Imbalance table in reports/task1b_imbalance_table.md.

Next: Deploy model in production
  