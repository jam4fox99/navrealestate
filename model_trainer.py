#!/usr/bin/env python3
"""
FHLB Loan Risk Model Trainer
Trains an ElasticNet logistic regression model for risk prediction
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

FEATURES = [
    'LTVRatioPercent',
    'Borrower1CreditScoreValue',
    'TotalDebtExpenseRatioPercent',
    'HousingExpenseRatioPercent',
    'NoteRatePercent',
    'LoanAcquisitionActualUPBAmt',
    'LoanAmortizationMaxTermMonths',
    'TotalMonthlyIncomeAmount',
    'Borrower1AgeAtApplicationYears',
]


def compute_risk_score(row):
    """Compute risk score based on LTV, credit score, and DTI"""
    ltv = row.get('LTVRatioPercent', 80)
    if ltv is None or pd.isna(ltv):
        ltv = 80
    ltv = float(ltv) if isinstance(ltv, str) else ltv
    ltv_score = min(ltv / 100, 1.5)
    
    credit = row.get('Borrower1CreditScoreValue', 680)
    if credit is None or pd.isna(credit) or credit in [9, 99]:
        credit = 680
    credit_norm = 1 - ((credit - 300) / 550)
    credit_norm = max(0, min(1, credit_norm))
    
    dti = row.get('TotalDebtExpenseRatioPercent', 35)
    if dti is None or pd.isna(dti):
        dti = 35
    dti = float(dti) if isinstance(dti, str) else dti
    dti_score = min(dti / 100, 1)
    
    risk = (0.4 * ltv_score + 0.35 * credit_norm + 0.25 * dti_score) * 100
    return round(risk, 2)


def fetch_data_from_supabase(sample_size=50000):
    """Fetch loan data from Supabase with pagination"""
    print("Connecting to Supabase...")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    columns = ','.join([
        'Year', 'LTVRatioPercent', 'Borrower1CreditScoreValue', 
        'TotalDebtExpenseRatioPercent', 'HousingExpenseRatioPercent',
        'NoteRatePercent', 'LoanAcquisitionActualUPBAmt',
        'LoanAmortizationMaxTermMonths', 'TotalMonthlyIncomeAmount',
        'Borrower1AgeAtApplicationYears', 'Bank', 'FIPSStateNumericCode'
    ])
    
    all_data = []
    batch_size = 1000
    offset = 0
    
    print(f"Fetching data in batches of {batch_size}...")
    
    while len(all_data) < sample_size:
        response = supabase.table('all_loans').select(columns).range(offset, offset + batch_size - 1).execute()
        
        if not response.data:
            break
            
        all_data.extend(response.data)
        offset += batch_size
        print(f"  Fetched {len(all_data)} records so far...")
        
        if len(response.data) < batch_size:
            break
    
    df = pd.DataFrame(all_data[:sample_size])
    print(f"Total fetched: {len(df)} records")
    
    return df


def prepare_features(df):
    """Prepare features for model training"""
    df = df.copy()
    
    for col in ['LTVRatioPercent', 'TotalDebtExpenseRatioPercent', 'HousingExpenseRatioPercent']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['LTVRatioPercent'] = df['LTVRatioPercent'].fillna(80)
    df['Borrower1CreditScoreValue'] = df['Borrower1CreditScoreValue'].replace([9, 99], np.nan).fillna(680)
    df['TotalDebtExpenseRatioPercent'] = df['TotalDebtExpenseRatioPercent'].fillna(35)
    df['HousingExpenseRatioPercent'] = df['HousingExpenseRatioPercent'].fillna(25)
    df['NoteRatePercent'] = df['NoteRatePercent'].fillna(5)
    df['LoanAcquisitionActualUPBAmt'] = df['LoanAcquisitionActualUPBAmt'].fillna(df['LoanAcquisitionActualUPBAmt'].median())
    df['LoanAmortizationMaxTermMonths'] = df['LoanAmortizationMaxTermMonths'].fillna(360)
    df['TotalMonthlyIncomeAmount'] = df['TotalMonthlyIncomeAmount'].fillna(df['TotalMonthlyIncomeAmount'].median())
    df['Borrower1AgeAtApplicationYears'] = df['Borrower1AgeAtApplicationYears'].replace([998, 999], np.nan).fillna(40)
    
    df['risk_score'] = df.apply(compute_risk_score, axis=1)
    
    # Use median as threshold to ensure balanced classes
    risk_threshold = df['risk_score'].median()
    df['high_risk'] = (df['risk_score'] > risk_threshold).astype(int)
    print(f"Risk threshold (median): {risk_threshold:.2f}")
    
    return df


def train_model(df):
    """Train the risk prediction model"""
    print("\nPreparing data for training...")
    
    available_features = [f for f in FEATURES if f in df.columns]
    print(f"Using features: {available_features}")
    
    X = df[available_features].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    y = df['high_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"High risk ratio: {y.mean():.2%}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nTraining Logistic Regression with L2 regularization...")
    model = LogisticRegression(
        solver='lbfgs',
        C=1.0,
        max_iter=1000,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    print("\n=== Model Performance ===")
    print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    print("\n=== Feature Importance ===")
    importance = pd.DataFrame({
        'feature': available_features,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)
    print(importance.to_string(index=False))
    
    return model, scaler, available_features


def save_model(model, scaler, features):
    """Save the trained model and scaler"""
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': features
    }
    
    model_path = os.path.join(models_dir, 'risk_model.pkl')
    joblib.dump(model_data, model_path)
    print(f"\nModel saved to {model_path}")


def main():
    df = fetch_data_from_supabase(sample_size=50000)
    
    if len(df) == 0:
        print("No data found in database.")
        return
    
    df = prepare_features(df)
    
    print(f"\nRisk score distribution:")
    print(df['risk_score'].describe())
    
    model, scaler, features = train_model(df)
    
    save_model(model, scaler, features)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
