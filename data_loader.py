#!/usr/bin/env python3
"""
FHLB Loan Data Loader
Loads CSV data from the data/ folder into Supabase
"""

import os
import pandas as pd
import numpy as np
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

OLD_SCHEMA_MAPPING = {
    'Year': 'year',
    'AssignedID': 'loan_id',
    'FHLBank': 'bank',
    'Bank': 'bank',
    'FIPSStateCode': 'state_code',
    'FIPSCountyCode': 'county_code',
    'MSA': 'cbsa_code',
    'Tract': 'census_tract',
    'MinPer': 'minority_ratio_pct',
    'TraMedY': 'tract_median_income',
    'LocMedY': 'local_median_income',
    'Income': 'monthly_income',
    'CurAreY': 'hud_median_income',
    'UPB': 'upb_amount',
    'LTV': 'ltv_ratio',
    'MortDate': 'note_date',
    'AcqDate': 'acquisition_date',
    'AcquDate': 'acquisition_date',
    'Purpose': 'loan_purpose',
    'Product': 'product_category',
    'FedGuar': 'mortgage_type',
    'Term': 'term_months',
    'AmorTerm': 'amort_term_months',
    'SellType': 'seller_type',
    'NumBor': 'borrower_count',
    'First': 'first_time_buyer',
    'BoRace': 'borrower1_race',
    'CoRace': 'borrower2_race',
    'BoSex': 'borrower1_sex',
    'CoSex': 'borrower2_sex',
    'BoAge': 'borrower1_age',
    'CoAge': 'borrower2_age',
    'Occup': 'property_usage',
    'NumUnits': 'property_units',
    'Rate': 'note_rate',
    'Amount': 'note_amount',
    'Front': 'housing_expense_ratio',
    'Back': 'total_debt_ratio',
    'BoCreditScor': 'borrower1_credit_score',
    'BoCreditScore': 'borrower1_credit_score',
    'CoCreditScor': 'borrower2_credit_score',
    'CoBoCreditScore': 'borrower2_credit_score',
    'CoCreditScore': 'borrower2_credit_score',
    'PMI': 'pmi_coverage',
    'Self': 'self_employed',
    'PropType': 'property_type',
}

NEW_SCHEMA_MAPPING = {
    'Year': 'year',
    'LoanCharacteristicsID': 'loan_id',
    'Bank': 'bank',
    'FIPSStateNumericCode': 'state_code',
    'FIPSCountyCode': 'county_code',
    'CoreBasedStatisticalAreaCode': 'cbsa_code',
    'CensusTractIdentifier': 'census_tract',
    'CensusTractMinorityRatioPercent': 'minority_ratio_pct',
    'CensusTractMedFamIncomeAmount': 'tract_median_income',
    'LocalAreaMedianIncomeAmount': 'local_median_income',
    'TotalMonthlyIncomeAmount': 'monthly_income',
    'HUDMedianIncomeAmount': 'hud_median_income',
    'LoanAcquisitionActualUPBAmt': 'upb_amount',
    'LTVRatioPercent': 'ltv_ratio',
    'NoteDate': 'note_date',
    'LoanAcquisitionDate': 'acquisition_date',
    'LoanAcquistionDate': 'acquisition_date',
    'LoanPurposeType': 'loan_purpose',
    'ProductCategoryName': 'product_category',
    'MortgageType': 'mortgage_type',
    'ScheduledTotalPaymentCount': 'term_months',
    'LoanAmortizationMaxTermMonths': 'amort_term_months',
    'MortgageLoanSellerInstType': 'seller_type',
    'BorrowerCount': 'borrower_count',
    'BorrowerFirstTimeHomebuyer': 'first_time_buyer',
    'Borrower1Race1Type': 'borrower1_race',
    'Borrower2Race1Type': 'borrower2_race',
    'Borrower1SexType': 'borrower1_sex',
    'Borrower2SexType': 'borrower2_sex',
    'Borrower1AgeAtApplicationYears': 'borrower1_age',
    'Borrower2AgeAtApplicationYears': 'borrower2_age',
    'PropertyUsageType': 'property_usage',
    'PropertyUnitCount': 'property_units',
    'NoteRatePercent': 'note_rate',
    'NoteAmount': 'note_amount',
    'HousingExpenseRatioPercent': 'housing_expense_ratio',
    'TotalDebtExpenseRatioPercent': 'total_debt_ratio',
    'Borrower1CreditScoreValue': 'borrower1_credit_score',
    'Borrower2CreditScoreValue': 'borrower2_credit_score',
    'PMICoveragePercent': 'pmi_coverage',
    'EmploymentBorrowerSelfEmployed': 'self_employed',
    'PropertyType': 'property_type',
}

TARGET_COLUMNS = [
    'year', 'loan_id', 'bank', 'state_code', 'county_code', 'cbsa_code',
    'census_tract', 'minority_ratio_pct', 'tract_median_income',
    'local_median_income', 'monthly_income', 'hud_median_income',
    'upb_amount', 'ltv_ratio', 'note_date', 'acquisition_date',
    'loan_purpose', 'product_category', 'mortgage_type', 'term_months',
    'amort_term_months', 'seller_type', 'borrower_count', 'first_time_buyer',
    'borrower1_race', 'borrower2_race', 'borrower1_sex', 'borrower2_sex',
    'borrower1_age', 'borrower2_age', 'property_usage', 'property_units',
    'note_rate', 'note_amount', 'housing_expense_ratio', 'total_debt_ratio',
    'borrower1_credit_score', 'borrower2_credit_score', 'pmi_coverage',
    'self_employed', 'property_type'
]


def detect_schema(df):
    """Detect if file uses old or new schema"""
    if 'LoanCharacteristicsID' in df.columns:
        return 'new'
    return 'old'


def normalize_columns(df):
    """Normalize column names to target schema"""
    schema_type = detect_schema(df)
    mapping = NEW_SCHEMA_MAPPING if schema_type == 'new' else OLD_SCHEMA_MAPPING
    
    rename_map = {}
    for col in df.columns:
        if col in mapping:
            rename_map[col] = mapping[col]
    
    df = df.rename(columns=rename_map)
    
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            df[col] = None
    
    return df[TARGET_COLUMNS]


def clean_data(df):
    """Clean and process the data"""
    missing_values = [998, 999, 9998, 9999, 99999, 9999999999]
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].replace(missing_values, np.nan)
    
    df['borrower1_age'] = df['borrower1_age'].replace([98, 998], np.nan)
    df['borrower2_age'] = df['borrower2_age'].replace([98, 998], np.nan)
    
    if df['ltv_ratio'].max() <= 2:
        df['ltv_ratio'] = df['ltv_ratio'] * 100
    if df['note_rate'].max() <= 1:
        df['note_rate'] = df['note_rate'] * 100
    if df['housing_expense_ratio'].max() <= 1:
        df['housing_expense_ratio'] = df['housing_expense_ratio'] * 100
    if df['total_debt_ratio'].max() <= 1:
        df['total_debt_ratio'] = df['total_debt_ratio'] * 100
    
    return df


def compute_risk_score(df):
    """Compute risk score based on LTV, credit score, and DTI"""
    df = df.copy()
    
    ltv_score = df['ltv_ratio'].fillna(80) / 100
    ltv_score = ltv_score.clip(0, 1.5)
    
    credit_score = df['borrower1_credit_score'].fillna(680)
    credit_score = credit_score.replace([9, 99], 680)
    credit_norm = 1 - ((credit_score - 300) / 550).clip(0, 1)
    
    dti_score = df['total_debt_ratio'].fillna(35) / 100
    dti_score = dti_score.clip(0, 1)
    
    df['risk_score'] = (0.4 * ltv_score + 0.35 * credit_norm + 0.25 * dti_score) * 100
    df['risk_score'] = df['risk_score'].round(2)
    
    return df


def load_csv_files(data_dir):
    """Load all CSV files from data directory"""
    all_data = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(data_dir, filename)
            print(f"Loading {filename}...")
            
            df = pd.read_csv(filepath, encoding='utf-8-sig', low_memory=False)
            df = normalize_columns(df)
            df = clean_data(df)
            all_data.append(df)
            print(f"  Loaded {len(df)} rows")
    
    combined = pd.concat(all_data, ignore_index=True)
    combined = compute_risk_score(combined)
    
    return combined


def upload_to_supabase(df, batch_size=1000):
    """Upload dataframe to Supabase in batches"""
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Convert all numeric columns to float (NUMERIC type in DB accepts floats)
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
    
    records = df.to_dict('records')
    total = len(records)
    
    print(f"\nUploading {total} records to Supabase...")
    
    for i in range(0, total, batch_size):
        batch = records[i:i+batch_size]
        try:
            supabase.table('fhlb_loans').insert(batch).execute()
            print(f"  Uploaded {min(i+batch_size, total)}/{total} records")
        except Exception as e:
            print(f"  Error at batch {i}: {e}")
            continue
    
    print("Upload complete!")


def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    df = load_csv_files(data_dir)
    print(f"\nTotal records loaded: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nRisk score stats:")
    print(df['risk_score'].describe())
    
    upload_to_supabase(df)


if __name__ == "__main__":
    main()
