
### 3. scripts/README.md (inside scripts folder)

```markdown
# Scripts Folder README

This folder contains standalone Python scripts for preprocessing and modeling.

### Available Scripts
- **task1_preprocessing.py**  
  Full Task 1 preprocessing pipeline (both datasets):  
  - Load raw data  
  - Cleaning  
  - Geolocation mapping (IP → Country)  
  - Feature engineering  
  - Transformation (StandardScaler + OneHotEncoder)  
  - Dense saving fix (.toarray()) – no sparse string issues  
  - Saves preprocessor & scaler

- **task2_modeling.py**  
  Standalone Task 2 script (matches modeling.ipynb):  
  - Load SMOTE train + original test  
  - Logistic Regression baseline + Random Forest champion  
  - Metrics (PR-AUC, F1), confusion matrix, CV, tuning  
  - Saves model & figures

### How to Run
```bash
# Run preprocessing (if re-generating data)
python scripts/task1_preprocessing.py

# Run modeling
python scripts/task2_modeling.py

Output:

Processed CSVs in data/processed/ (dense numeric format)
Confusion matrix in reports/figures/
Results in reports/task2.log
Important: Re-run preprocessing before modeling if data changes.