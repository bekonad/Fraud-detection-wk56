# scripts/task1_preprocessing.py
# Task 1: Data Analysis and Preprocessing – Complete & Stable Version
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
import os
import joblib

# Setup
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("Task 1 Preprocessing Starting...")

# 1. Load Data
def load_data():
    fraud_data = pd.read_csv("data/raw/Fraud_Data.csv")
    ip_to_country = pd.read_csv("data/raw/IpAddress_to_Country.csv")
    creditcard_data = pd.read_csv("data/raw/creditcard.csv")
    return fraud_data, ip_to_country, creditcard_data

# 2. Clean Data
def clean_data(fraud_data, creditcard_data):
    fraud_data = fraud_data.dropna().drop_duplicates()
    fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
    fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
    fraud_data['ip_address'] = fraud_data['ip_address'].astype('int64')
    
    creditcard_data = creditcard_data.dropna().drop_duplicates()
    return fraud_data, creditcard_data

# 3. Geolocation Mapping
def merge_geolocation(fraud_data, ip_to_country):
    ip_to_country = ip_to_country.sort_values('lower_bound_ip_address').reset_index(drop=True)
    
    def map_ip(ip):
        idx = np.searchsorted(ip_to_country['lower_bound_ip_address'], ip, side='right') - 1
        if idx >= 0 and ip <= ip_to_country['upper_bound_ip_address'].iloc[idx]:
            return ip_to_country['country'].iloc[idx]
        return 'Unknown'
    
    fraud_data['country'] = fraud_data['ip_address'].apply(map_ip)
    return fraud_data

# 4. Feature Engineering
def feature_engineering(fraud_data):
    fraud_data['time_since_signup_hours'] = (fraud_data['purchase_time'] - fraud_data['signup_time']).dt.total_seconds() / 3600
    fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
    fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
    
    user_freq = fraud_data['user_id'].value_counts()
    fraud_data['transaction_frequency'] = fraud_data['user_id'].map(user_freq)
    
    fraud_data['velocity'] = fraud_data['purchase_value'] / (fraud_data['time_since_signup_hours'] + 1)
    
    high_risk = ['Luxembourg', 'Ecuador', 'Tunisia', 'Peru', 'Bolivia']
    fraud_data['high_risk_country'] = fraud_data['country'].isin(high_risk).astype(int)
    
    return fraud_data

# 5. Data Transformation (No SMOTE — use class_weight in modeling)
def transform_data(fraud_data, creditcard_data):
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
    
    # Credit card (only scaling)
    scaler = StandardScaler()
    X_cc = creditcard_data.drop('Class', axis=1)
    y_cc = creditcard_data['Class']
    
    X_cc_train, X_cc_test, y_cc_train, y_cc_test = train_test_split(
        X_cc, y_cc, test_size=0.2, random_state=42, stratify=y_cc)
    
    X_cc_train_trans = scaler.fit_transform(X_cc_train)
    X_cc_test_trans = scaler.transform(X_cc_test)
    
    # Save
    pd.DataFrame(X_fraud_train_trans).to_csv("data/processed/X_fraud_train.csv", index=False)
    pd.DataFrame(X_fraud_test_trans).to_csv("data/processed/X_fraud_test.csv", index=False)
    pd.Series(y_fraud_train).to_csv("data/processed/y_fraud_train.csv", index=False)
    pd.Series(y_fraud_test).to_csv("data/processed/y_fraud_test.csv", index=False)
    
    pd.DataFrame(X_cc_train_trans).to_csv("data/processed/X_creditcard_train.csv", index=False)
    pd.DataFrame(X_cc_test_trans).to_csv("data/processed/X_creditcard_test.csv", index=False)
    pd.Series(y_cc_train).to_csv("data/processed/y_creditcard_train.csv", index=False)
    pd.Series(y_cc_test).to_csv("data/processed/y_creditcard_test.csv", index=False)
    
    joblib.dump(preprocessor, "data/processed/preprocessor.pkl")
    joblib.dump(scaler, "data/processed/scaler.pkl")
    
    print("Task 1 complete — no SMOTE dependency, using class_weight='balanced' in modeling")

# Main
def main():
    fraud_data, ip_to_country, creditcard_data = load_data()
    fraud_data, creditcard_data = clean_data(fraud_data, creditcard_data)
    fraud_data = merge_geolocation(fraud_data, ip_to_country)
    fraud_data = feature_engineering(fraud_data)
    transform_data(fraud_data, creditcard_data)

if __name__ == "__main__":
    main()