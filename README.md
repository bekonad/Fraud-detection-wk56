# Final Project Report  
**Improved Detection of Fraud Cases for E-Commerce and Bank Transactions**  
**10 Academy Artificial Intelligence Mastery – Week 5&6 Challenge**  
**Author:** Bereket Feleke  
**GitHub Repository:** https://github.com/bekonad/Fraud-detection-wk56  
**Date:** December 30, 2025

## Executive Summary

This project delivers an end-to-end fraud detection system for e-commerce (`Fraud_Data.csv`) and bank transactions (`creditcard.csv`) using machine learning. The focus was to achieve high fraud detection accuracy while minimizing false positives (customer friction) in highly imbalanced datasets.

**Key Results**:
- **Task 1**: Comprehensive EDA, geolocation mapping, feature engineering (velocity, time_since_signup_hours), SMOTE imbalance handling (50/50 train)
- **Task 2**: Random Forest champion model (PR-AUC 0.8990, F1 0.7021, **only 12 false positives**)
- **Task 3**: SHAP explainability – top drivers: velocity, time_since_signup_hours, high_risk_country
- **Business Impact**: Extremely low false positives → excellent customer experience; strong fraud capture → significant loss reduction

**Champion Model**: Random Forest – selected for superior performance, robustness, and interpretability via SHAP.

## Project Overview & Business Understanding

**Challenge Objective**: Build interpretable, high-performance fraud detection models for e-commerce and bank transactions using alternative data (behavioral, geolocation, time patterns).

**Business Goal**:
- Reduce fraud losses while maintaining excellent customer experience
- Balance false positives (customer friction) and false negatives (missed fraud)

**Datasets**:
- `Fraud_Data.csv`: E-commerce transactions (behavioral, geolocation, purchase patterns)
- `creditcard.csv`: Anonymized bank transactions (PCA features, extreme imbalance ~0.17%)

**Key Innovation**: Geolocation mapping (IP → Country), transaction velocity, time_since_signup_hours → detect fraud without traditional labels.

## Repository Structure

Fraud-detection-wk56/
├── data/
│   ├── raw/                  # Original datasets
│   └── processed/            # SMOTE train, original test (dense)
├── notebooks/
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb
│   └── shap_explainability.ipynb
├── reports/
│   ├── figures/              # Confusion matrix, feature importance, SHAP plots
│   └── task1b_imbalance_table.md
├── scripts/
│   ├── task1_preprocessing.py
│   └── task2_modeling.py
├── README.md
└── requirements.txt


## Task 1: Data Analysis & Preprocessing

**1a – Cleaning & EDA**
- Explicit cleaning: datetime conversion, type casting, no missing/duplicates
- Univariate/Bivariate: Amount, Time, V-features distributions (log-scale), geolocation mapping
- Imbalance: ~9.39% fraud in Fraud_Data, ~0.173% in creditcard
- Visualizations: saved to `reports/figures/`

**1b – Feature Engineering, Transformation & Imbalance**
- Engineered features: time_since_signup_hours, hour_of_day, day_of_week, velocity
- Pipeline: StandardScaler + OneHotEncoder → dense arrays saved
- SMOTE: applied only on train → 50/50 balance
- Imbalance table: saved to `reports/task1b_imbalance_table.md`

**Key Insights**:
- Fraud occurs quickly after signup (same day)
- High-risk countries (Luxembourg, Ecuador) and device sharing are strong signals

## Task 2: Model Building & Evaluation

**Baseline**: Logistic Regression
- PR-AUC: 0.6398
- F1-Score: 0.6756

**Champion**: Random Forest
- PR-AUC: 0.8990
- F1-Score: 0.7021
- False Positives: **12** (excellent customer experience)
- Confusion Matrix: saved to `reports/figures/random_forest_confusion_matrix.png`

**Cross-Validation & Tuning**:
- 5-Fold CV: Mean F1 0.7007
- GridSearchCV: optimized n_estimators & max_depth

**Model Comparison Table**:

| Model              | PR-AUC | F1-Score | Mean CV F1 |
|--------------------|--------|----------|------------|
| Logistic Regression | 0.6398 | 0.6756   | -          |
| Random Forest      | 0.8990 | 0.7021   | 0.7007     |

**Conclusion**: Random Forest is the champion – high precision, minimal false alarms.

## Task 3: Model Explainability – SHAP Analysis

**Champion Model**: Random Forest

**1. Built-in Feature Importance (Top 10)**

Top 10 Built-in Feature Importance (Random Forest):
| Feature      | Importance |
|--------------|------------|
| feature_2    | 0.392834   |
| feature_3    | 0.360097   |
| feature_1    | 0.0320092  |
| feature_0    | 0.0284065  |
| feature_179  | 0.0280703  |
| feature_5    | 0.019385   |
| feature_10   | 0.0181987  |
| feature_4    | 0.0154586  |
| feature_7    | 0.0117991  |
| feature_9    | 0.00924116 |

**Visualization**: `reports/figures/built_in_feature_importance.png`

**2. SHAP Summary Plot (Global Importance)**

SHAP Summary Plot – Fraud Class:  
(High feature value in red → increases fraud probability)  
Top drivers: velocity (high value = higher fraud risk), time_since_signup_hours (short time = higher risk), high_risk_country.

**Visualization**: `reports/figures/shap_summary_fraud.png`

**3. SHAP Force Plots (Individual Predictions)**

- True Positive (correct fraud): velocity + short signup time push prediction to fraud  
  → `reports/figures/shap_force_true_positive.png`

- False Positive (legitimate flagged): high_risk_country + high velocity caused misflag  
  → `reports/figures/shap_force_false_positive.png`

- False Negative (missed fraud): low velocity + long signup time hid fraud  
  → `reports/figures/shap_force_false_negative.png`

**4. Interpretation**

- **Top 5 Fraud Drivers** (SHAP + built-in):
  1. velocity
  2. time_since_signup_hours
  3. high_risk_country
  4. hour_of_day
  5. purchase_value

- **SHAP vs Built-in**: SHAP better captures interactions (e.g., high_risk_country stronger in SHAP)

- **Surprising findings**: High velocity from high-risk countries amplified in SHAP (strong interaction effect)

**5. Business Recommendations**

1. **Immediate OTP/2FA for high-risk patterns**  
   Transactions within 24 hours of signup + high velocity → require OTP/2FA verification (SHAP shows these push fraud probability high).

2. **Enhanced KYC for geolocation risk**  
   High-risk country (e.g., Luxembourg, Ecuador) + short signup time → mandatory KYC/ID check (geolocation is top driver in SHAP).

3. **Manual review for suspicious patterns**  
   Mid-range purchase value + off-hour time → flag for manual review (patterns in EDA & SHAP).

## Conclusion & Business Impact

**Achievements**:
- End-to-end pipeline: cleaning → EDA → features → SMOTE → modeling → SHAP
- Outstanding model: Random Forest with only **12 false positives** (minimal customer friction)
- Explainable insights: velocity and signup time are critical fraud signals

**Business Impact**:
- Low false positives → excellent customer experience
- High fraud capture → significant loss reduction

**Limitations**:
- Creditcard modeling pending (feature mismatch) – focus was Fraud_Data
- SHAP on 100 samples (full dataset would be more robust but slow)

**Next Steps**:
- Deploy model via API (FastAPI + Docker)
- Real-time monitoring dashboard
- Continuous model retraining

**Reproducibility**:
```bash
git clone https://github.com/bekonad/Fraud-detection-wk56.git
cd Fraud-detection-wk56
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook

Prepared by: Bereket Feleke
Date: December 30, 2025