# scripts/task2_modeling.py
# Task 2: Model Building and Evaluation – Standalone Script
# Author: Bereket Feleke
# Date: 28 December 2025
# Aligned with modeling.ipynb (SMOTE train, original test, RF champion, 12 FP, CV F1 ~0.7007)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, f1_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold

# Setup
np.random.seed(42)
os.makedirs("reports/models", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("reports/task2.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_processed_data():
    """Load SMOTE train and original test from data/processed/."""
    logger.info("Loading SMOTE processed data from data/processed/...")
    base_path = "data/processed/"  # Correct path from project root
    try:
        X_train = pd.read_csv(base_path + "X_fraud_train_smote.csv", header=None)
        y_train = pd.read_csv(base_path + "y_fraud_train_smote.csv", header=None).values.ravel()
        X_test = pd.read_csv(base_path + "X_fraud_test.csv", header=None)
        y_test = pd.read_csv(base_path + "y_fraud_test.csv", header=None).values.ravel()

        # Convert any object columns to numeric (fixes sparse/string issue)
        for df in [X_train, X_test]:
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        logger.info("Data loaded and converted to dense numeric format.")
        return X_train.to_numpy(), y_train, X_test.to_numpy(), y_test
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Run feature-engineering.ipynb first to generate SMOTE files")
        raise

def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    """Train model, evaluate, save confusion matrix."""
    logger.info(f"Training {model_name}...")
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    f1 = f1_score(y_test, y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Legitimate', 'Fraud'])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.savefig(f"reports/figures/cm_{model_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"{model_name} → PR-AUC: {pr_auc:.4f}, F1: {f1:.4f}")
    logger.info(f"Classification Report:\n{classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud'])}")

    return pr_auc, f1, model

def cross_validation(model, X_train, y_train):
    logger.info("Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    logger.info(f"Mean CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    return cv_scores.mean()

def hyperparameter_tuning(model, X_train, y_train):
    logger.info("Running hyperparameter tuning (GridSearchCV)...")
    param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
    grid = GridSearchCV(model, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    logger.info(f"Best parameters: {grid.best_params_}")
    logger.info(f"Best CV F1 score: {grid.best_score_:.4f}")
    return grid.best_score_

def model_comparison(pr_auc_lr, f1_lr, pr_auc_rf, f1_rf, cv_mean):
    logger.info("\n=== Model Comparison Table ===")
    comparison = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest'],
        'PR-AUC': [pr_auc_lr, pr_auc_rf],
        'F1-Score': [f1_lr, f1_rf],
        'Mean CV F1': [np.nan, cv_mean]
    })
    logger.info(comparison.to_markdown(index=False))

def main():
    logger.info("Starting Task 2: Model Building...")
    try:
        X_train, y_train, X_test, y_test = load_processed_data()

        # Baseline: Logistic Regression
        logreg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        pr_auc_lr, f1_lr, _ = train_and_evaluate(logreg, X_train, X_test, y_train, y_test, "LogisticRegression")

        # Champion: Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1)
        pr_auc_rf, f1_rf, rf_model = train_and_evaluate(rf, X_train, X_test, y_train, y_test, "RandomForest")

        # Cross-Validation on champion
        cv_mean = cross_validation(rf_model, X_train, y_train)

        # Hyperparameter tuning on champion
        hyperparameter_tuning(rf_model, X_train, y_train)

        # Comparison table
        model_comparison(pr_auc_lr, f1_lr, pr_auc_rf, f1_rf, cv_mean)

        logger.info("Task 2 complete – Random Forest champion selected (low FP, high precision)")
        logger.info("Next: SHAP explainability (Task 3)")

    except Exception as e:
        logger.error(f"Critical error: {e}")
        raise

if __name__ == "__main__":
    main()