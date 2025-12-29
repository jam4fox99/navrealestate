# FHLB Loan Risk Calculator

A Streamlit web application for predicting loan default risk based on Federal Home Loan Bank (FHLB) Public Use Database data.

## Features

- **Risk Calculator**: Input loan parameters (LTV, credit score, DTI, etc.) and get instant default probability predictions
- **Risk Gauge**: Visual gauge showing risk level (Low/Medium/High/Very High)
- **Sensitivity Analysis**: See how changing one variable affects the risk score
- **Database Insights**: View high-risk loans, distributions, and yearly trends

## Data

- **Source**: FHLB Public Use Database (2021-2024)
- **Records**: 178,162 loans
- **Storage**: Supabase PostgreSQL

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment**:
   Copy `.env.example` to `.env` and add your Supabase credentials:
   ```
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_SERVICE_ROLE_KEY=your-key
   ```

3. **Train the model** (if not already trained):
   ```bash
   python model_trainer.py
   ```

4. **Run the app**:
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
Nav Realestate/
├── app.py              # Streamlit frontend
├── model_trainer.py    # ML training script
├── data_loader.py      # CSV to Supabase loader
├── requirements.txt    # Python dependencies
├── .env.example        # Environment template
├── .gitignore
├── models/
│   └── risk_model.pkl  # Trained model
└── data/               # CSV files (gitignored)
```

## Risk Model

- **Algorithm**: Logistic Regression with L2 regularization
- **Features**: LTV ratio, Credit Score, DTI, Housing Expense Ratio, Interest Rate, Loan Amount, Term, Monthly Income, Borrower Age
- **Risk Score**: Weighted combination of LTV (40%), Credit Score (35%), DTI (25%)

## Limitations

- Risk model is based on loan characteristics only, not actual default outcomes
- Historical default data would improve prediction accuracy
- Model should be retrained periodically as new data becomes available
