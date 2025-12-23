# src/task1_preprocessing.py
# Task 1: Data Analysis and Preprocessing – Complete Implementation
# Adey Innovations Inc. – 10 Academy Week 5&6 Fraud Detection Challenge
# Author: Bereket Feleke
# Date: 23 December 2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import os
import joblib
import logging

# Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("reports/task1.log"), logging.StreamHandler()]
)

np.random.seed(42)
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("reports/models", exist_ok=True)

# 1. Load Data
def load_data():
    logging.info("Loading raw datasets...")
    fraud_data = pd.read_csv("data/raw/Fraud_Data.csv")
    ip_to_country = pd.read_csv("data/raw/IpAddress_to_Country.csv")
    creditcard_data = pd.read_csv("data/raw/creditcard.csv")
    logging.info("Datasets loaded successfully.")
    return fraud_data, ip_to_country, creditcard_data

# 2. Clean Data
def clean_data(fraud_data, creditcard_data):
    logging.info("Cleaning data...")
    
    # Fraud data
    fraud_data = fraud_data.dropna().drop_duplicates()
    fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
    fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
    fraud_data['ip_address'] = fraud_data['ip_address'].astype('int64')
    
    # Credit card data
    creditcard_data = creditcard_data.dropna().drop_duplicates()
    
    logging.info("Data cleaning complete.")
    return fraud_data, creditcard_data

# 3. Geolocation Mapping
def merge_geolocation(fraud_data, ip_to_country):
    logging.info("Starting geolocation mapping...")
    
    ip_to_country = ip_to_country.sort_values('lower_bound_ip_address').reset_index(drop=True)
    
    def map_ip(ip):
        idx = np.searchsorted(ip_to_country['lower_bound_ip_address'], ip, side='right') - 1
        if idx >= 0 and ip <= ip_to_country['upper_bound_ip_address'].iloc[idx]:
            return ip_to_country['country'].iloc[idx]
        return 'Unknown'
    
    fraud_data['country'] = fraud_data['ip_address'].apply(map_ip)
    
    logging.info(f"Mapping complete. Unique countries: {fraud_data['country'].nunique()}")
    return fraud_data

# 4. Feature Engineering
def feature_engineering(fraud_data):
    logging.info("Feature engineering...")
    
    fraud_data['time_since_signup_hours'] = (fraud_data['purchase_time'] - fraud_data['signup_time']).dt.total_seconds() / 3600
    fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
    fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
    
    # Transaction frequency per user
    user_freq = fraud_data['user_id'].value_counts()
    fraud_data['transaction_frequency'] = fraud_data['user_id'].map(user_freq)
    
    # Velocity (amount per hour since signup)
    fraud_data['velocity'] = fraud_data['purchase_value'] / (fraud_data['time_since_signup_hours'] + 1)
    
    # High-risk country flag (from EDA)
    high_risk = ['Luxembourg', 'Ecuador', 'Tunisia', 'Peru', 'Bolivia']
    fraud_data['high_risk_country'] = fraud_data['country'].isin(high_risk).astype(int)
    
    logging.info("Feature engineering complete.")
    return fraud_data

# 5. EDA Visualizations (with saving)
def save_plot(fig, filename):
    path = f"reports/figures/{filename}"
    fig.savefig(path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    logging.info(f"Plot saved: {path}")

def perform_eda(fraud_data, creditcard_data):
    logging.info("Generating EDA visualizations...")
    
    # Class imbalance - Fraud_Data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    sns.countplot(x='class', data=fraud_data, ax=ax1)
    ax1.set_title('Fraud_Data Class Distribution')
    fraud_data['class'].value_counts().plot.pie(ax=ax2, autopct='%1.2f%%')
    ax2.set_title('Fraud Proportion')
    save_plot(fig, "fraud_class_distribution.png")
    
    # Country fraud rate
    country_stats = fraud_data.groupby('country')['class'].agg(['count', 'mean'])
    country_stats = country_stats[country_stats['count'] >= 50].sort_values('mean', ascending=False).head(15)
    fig = plt.figure()
    sns.barplot(x='mean', y=country_stats.index, data=country_stats.reset_index(), palette='Reds_r')
    plt.title('Top High-Risk Countries')
    save_plot(fig, "high_risk_countries.png")
    
    # Credit card imbalance
    fig = plt.figure()
    sns.countplot(x='Class', data=creditcard_data)
    plt.title('Credit Card Class Distribution')
    save_plot(fig, "creditcard_class_distribution.png")
    
    logging.info("All EDA plots saved in reports/figures/")

# 6. Data Transformation & SMOTE
def transform_and_resample(fraud_data, creditcard_data):
    logging.info("Data transformation and resampling...")
    
    # Fraud data features
    num_features = ['purchase_value', 'age', 'time_since_signup_hours', 'velocity', 'transaction_frequency']
    cat_features = ['source', 'browser', 'sex', 'country']
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
    ])
    
    X_fraud = fraud_data[num_features + cat_features]
    y_fraud = fraud_data['class']
    
    X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(
        X_fraud, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud)
    
    X_fraud_train_trans = preprocessor.fit_transform(X_fraud_train)
    X_fraud_test_trans = preprocessor.transform(X_fraud_test)
    
    # SMOTE
    print("Before SMOTE:", Counter(y_fraud_train))
    smote = SMOTE(random_state=42)
    X_fraud_train_res, y_fraud_train_res = smote.fit_resample(X_fraud_train_trans, y_fraud_train)
    print("After SMOTE:", Counter(y_fraud_train_res))
    
    # Credit card data (only numerical)
    scaler = StandardScaler()
    X_cc = creditcard_data.drop('Class', axis=1)
    y_cc = creditcard_data['Class']
    
    X_cc_train, X_cc_test, y_cc_train, y_cc_test = train_test_split(
        X_cc, y_cc, test_size=0.2, random_state=42, stratify=y_cc)
    
    X_cc_train_trans = scaler.fit_transform(X_cc_train)
    X_cc_test_trans = scaler.transform(X_cc_test)
    
    # Save everything
    pd.DataFrame(X_fraud_train_res).to_csv("data/processed/X_fraud_train.csv", index=False)
    pd.DataFrame(X_fraud_test_trans).to_csv("data/processed/X_fraud_test.csv", index=False)
    pd.Series(y_fraud_train_res).to_csv("data/processed/y_fraud_train.csv", index=False)
    pd.Series(y_fraud_test).to_csv("data/processed/y_fraud_test.csv", index=False)
    
    pd.DataFrame(X_cc_train_trans).to_csv("data/processed/X_creditcard_train.csv", index=False)
    pd.DataFrame(X_cc_test_trans).to_csv("data/processed/X_creditcard_test.csv", index=False)
    pd.Series(y_cc_train).to_csv("data/processed/y_creditcard_train.csv", index=False)
    pd.Series(y_cc_test).to_csv("data/processed/y_creditcard_test.csv", index=False)
    
    joblib.dump(preprocessor, "data/processed/preprocessor.pkl")
    joblib.dump(scaler, "data/processed/scaler.pkl")
    
    logging.info("All processed data and transformers saved.")

# Main
def main():
    fraud_data, ip_to_country, creditcard_data = load_data()
    fraud_data, creditcard_data = clean_data(fraud_data, creditcard_data)
    fraud_data = merge_geolocation(fraud_data, ip_to_country)
    fraud_data = feature_engineering(fraud_data)
    perform_eda(fraud_data, creditcard_data)
    transform_and_resample(fraud_data, creditcard_data)
    logging.info("Task 1 complete — ready for modeling!")

if __name__ == "__main__":
    main()