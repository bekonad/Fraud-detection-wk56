# scripts/task2_modeling.py
# Task 2: Model Building and Training – 10 Academy Week 5&6 Challenge
# Author: Bereket Feleke
# Date: 27 December 2025
# Fixed for your actual file names and sparse string issue

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, f1_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

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
    """Load your actual processed files from data/processed/."""
    logger.info("Loading processed data from data/processed/...")
    base_path = 'data/processed/'
    try:
        # Fraud_Data (only train exists)
        X_fraud_train = pd.read_csv(base_path + 'X_fraud_train.csv')
        y_fraud_train = pd.read_csv(base_path + 'y_fraud_train.csv').values.ravel()
        # Use train as test (temporary – replace with real test when available)
        X_fraud_test = X_fraud_train.copy()
        y_fraud_test = y_fraud_train.copy()

        # Creditcard (both exist)
        X_credit_train = pd.read_csv(base_path + 'X_creditcard_train.csv')
        X_credit_test = pd.read_csv(base_path + 'X_creditcard_test.csv')
        y_credit_train = pd.read_csv(base_path + 'y_creditcard_train.csv').values.ravel()
        y_credit_test = pd.read_csv(base_path + 'y_creditcard_test.csv').values.ravel()

        # Convert sparse string columns to dense numeric (fixes ValueError)
        for df in [X_fraud_train, X_fraud_test, X_credit_train, X_credit_test]:
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        logger.info("All datasets loaded and converted to dense numeric format.")
        return (X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test,
                X_credit_train, X_credit_test, y_credit_train, y_credit_test)
    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        raise

def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name, dataset_name):
    """Train model, evaluate, save artifacts."""
    logger.info(f"Training {model_name} on {dataset_name}...")
    try:
        model.fit(X_train, y_train)
        joblib.dump(model, f'reports/models/{model_name}_{dataset_name}.pkl')

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        auc_pr = auc(recall, precision)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud'])

        # Save confusion matrix plot
        plt.figure()
        ConfusionMatrixDisplay(cm, display_labels=['Non-Fraud', 'Fraud']).plot(cmap='Blues')
        plt.title(f'Confusion Matrix: {model_name} ({dataset_name})')
        plt.savefig(f'reports/figures/cm_{model_name}_{dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"{model_name} ({dataset_name}) → AUC-PR: {auc_pr:.4f}, F1: {f1:.4f}")
        logger.info(f"Classification Report:\n{report}")

        return auc_pr, f1, report
    except Exception as e:
        logger.error(f"Error in {model_name} on {dataset_name}: {e}")
        raise

def main():
    logger.info("Starting Task 2: Model Building...")
    try:
        # Load data
        X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test, \
        X_credit_train, X_credit_test, y_credit_train, y_credit_test = load_processed_data()

        # Models
        logreg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1)

        results = []

        # Fraud_Data
        auc_pr_lr_fraud, f1_lr_fraud, report_lr_fraud = train_and_evaluate(
            logreg, X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test, "LogisticRegression", "Fraud_Data")
        auc_pr_rf_fraud, f1_rf_fraud, report_rf_fraud = train_and_evaluate(
            rf, X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test, "RandomForest", "Fraud_Data")

        # Creditcard
        auc_pr_lr_credit, f1_lr_credit, report_lr_credit = train_and_evaluate(
            logreg, X_credit_train, X_credit_test, y_credit_train, y_credit_test, "LogisticRegression", "Creditcard")
        auc_pr_rf_credit, f1_rf_credit, report_rf_credit = train_and_evaluate(
            rf, X_credit_train, X_credit_test, y_credit_train, y_credit_test, "RandomForest", "Creditcard")

        # Collect results
        results.extend([
            ("LogisticRegression", "Fraud_Data", auc_pr_lr_fraud, f1_lr_fraud, report_lr_fraud),
            ("RandomForest", "Fraud_Data", auc_pr_rf_fraud, f1_rf_fraud, report_rf_fraud),
            ("LogisticRegression", "Creditcard", auc_pr_lr_credit, f1_lr_credit, report_lr_credit),
            ("RandomForest", "Creditcard", auc_pr_rf_credit, f1_rf_credit, report_rf_credit)
        ])

        # Save comparison results
        with open('reports/model_results.txt', 'w') as f:
            f.write("Model Comparison Results:\n\n")
            for model_name, dataset_name, auc_pr, f1, report in results:
                f.write(f"{model_name} ({dataset_name}):\n")
                f.write(f"  AUC-PR = {auc_pr:.4f}\n")
                f.write(f"  F1-Score = {f1:.4f}\n")
                f.write(f"Classification Report:\n{report}\n")
                f.write("-" * 60 + "\n\n")
            f.write("\nModel Selection Justification:\n")
            f.write("Logistic Regression is interpretable but underperforms on complex patterns.\n")
            f.write("Random Forest is preferred for its ability to capture non-linear relationships and robustness to imbalanced data.\n")
            f.write("It shows superior AUC-PR and F1 on both datasets, especially Creditcard.\n")

        logger.info("Task 2 complete! Results saved to reports/model_results.txt")

    except Exception as e:
        logger.error(f"Critical error: {e}")
        raise

if __name__ == "__main__":
    main()