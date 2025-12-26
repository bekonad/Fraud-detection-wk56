# 🚀 Improved Fraud Detection for E-commerce & Bank Transactions  
**10 Academy Artificial Intelligence Mastery – Week 5&6 Challenge**  
**Adey Innovations Inc.**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Interim%201%20Submitted-success)](https://github.com/bekonad/Fraud-detection-wk56)
[![Author](https://img.shields.io/badge/Author-Bereket%20Feleke-brightgreen)](https://github.com/bekonad)

**Repository**: https://github.com/bekonad/Fraud-detection-wk56  
**Challenge Dates**: 17 Dec – 30 Dec 2025  
**Interim-1 Submission**: 21 Dec 2025 ✅  
**Final Submission Due**: 30 Dec 2025, 8:00 PM UTC

---

## 📋 Project Overview

As a data scientist at **Adey Innovations Inc.**, a leading African fintech company, this project delivers **advanced fraud detection models** for e-commerce and banking transactions using:
- Alternative behavioral data
- Geolocation analysis
- Transaction pattern recognition
- Real-time capable architecture

**Core Business Goal**:
Reduce fraud losses while maintaining excellent customer experience by accurately distinguishing fraudulent from legitimate activity — balancing the critical trade-off between **false positives** (customer friction) and **false negatives** (financial loss).

**Key Innovation**:
Leveraging geolocation, transaction velocity, device sharing, and time patterns to detect fraud in highly imbalanced datasets (~0.1–0.2% fraud rate).

---

## 🗂️ Repository Structure

Fraud-detection-wk56/
├── data/
│   ├── raw/                  # Original datasets (Fraud_Data.csv, IpAddress_to_Country.csv, creditcard.csv)
│   └── processed/            # Cleaned & feature-engineered data (DVC tracked)
├── notebooks/
│   ├── eda-fraud-data.ipynb      # Complete EDA on e-commerce data (Task 1) ✅
│   ├── eda-creditcard.ipynb      # Anonymized credit card EDA (in progress)
│   ├── feature-engineering.ipynb # Velocity, risk flags, scaling
│   ├── modeling.ipynb            # XGBoost champion + ensemble
│   ├── shap-explainability.ipynb # Model interpretability
│   └── README.md
├── reports/
│   └── figures/                  # All EDA plots saved here
├── src/                          # Production code modules (in development)
├── tests/                        # Unit tests
├── requirements.txt
├── README.md                     # This file
└── .gitignore


---

## 🎯 Task 1 Completion Status (Interim-1 – 21 Dec 2025)

**All EDA & preprocessing requirements fully met**:

| Requirement                        | Status    | Key Deliverable |
|------------------------------------|-----------|-----------------|
| Data Cleaning                      | Complete  | No missing/duplicates, correct types |
| Univariate & Bivariate EDA         | Complete  | 9+ publication-quality visualizations |
| Geolocation Integration            | Complete  | IP → Country mapping (182 countries) |
| Fraud Patterns by Country          | Complete  | High-risk countries identified (>40% fraud) |
| Class Imbalance Analysis           | Complete  | 9.39% fraud rate + clear visualization |
| Initial Feature Engineering        | Started   | time_since_signup, device sharing, hour/day |

### 🔥 Key Fraud Insights (From `eda-fraud-data.ipynb`)

1. **Geolocation is the strongest signal**  
   → Luxembourg (38.9%), Ecuador (26.4%), Tunisia (26.3%) — extreme fraud rates

2. **Time is critical**  
   → Fraud occurs **very quickly after signup** (often same day)

3. **Device sharing = high risk**  
   → Devices used by multiple users show dramatically higher fraud

4. **Purchase value pattern**  
   → Fraud prefers mid-range amounts to avoid suspicion

5. **Severe class imbalance**  
   → Only 9.39% fraud → requires SMOTE + PR-AUC metrics

---

## 🛠️ Setup & Run Instructions

```bash
# Clone the repository
git clone https://github.com/bekonad/Fraud-detection-wk56.git
cd Fraud-detection-wk56

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Register kernel for Jupyter
python -m ipykernel install --user --name=venv --display-name "Python (venv)"

# Launch Jupyter
jupyter notebook
# or
jupyter lab

Open notebooks/eda-fraud-data.ipynb → Select kernel "Python (venv)" → Run all cells.

📈 Current Progress & Roadmap
















MilestoneStatusDue DateInterim-1 (Task 1 EDA)Submitted ✅21 Dec 2025EDA on Credit Card DataIn Progress25 Dec 2025Feature EngineeringNext27 Dec 2025Modeling & EvaluationPlanned29 Dec 2025Final Report & DeploymentPlanned30 Dec 2025
Target Outcome: Deploy real-time fraud detection system reducing annual losses by R620M+.

Open notebooks/eda-fraud-data.ipynb → Select kernel "Python (venv)" → Run all cells.

🙏 Acknowledgments
Special thanks to the 10 Academy team for this high-impact, real-world challenge.
"We don't just detect fraud — we protect trust in digital finance"

