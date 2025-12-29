#!/usr/bin/env python3
"""
FHLB Loan Risk Calculator - Professional UI
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

st.set_page_config(
    page_title="FHLB Loan Risk Calculator",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional look
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Sans:wght@400;500;700&display=swap');
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Root variables */
    :root {
        --primary: #0a0a0f;
        --secondary: #12121a;
        --card-bg: #1a1a24;
        --card-border: #2a2a3a;
        --accent: #6366f1;
        --accent-glow: rgba(99, 102, 241, 0.3);
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --text-primary: #f1f1f1;
        --text-secondary: #a1a1aa;
        --text-muted: #71717a;
    }
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, var(--primary) 0%, #0d0d14 100%);
        font-family: 'DM Sans', sans-serif;
    }
    
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Custom header */
    .custom-header {
        background: linear-gradient(90deg, var(--card-bg) 0%, var(--secondary) 100%);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .header-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.75rem;
        font-weight: 700;
        background: linear-gradient(135deg, #fff 0%, #a1a1aa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        color: var(--text-secondary);
        font-size: 0.875rem;
        margin-top: 0.25rem;
    }
    
    .header-badge {
        background: linear-gradient(135deg, var(--accent) 0%, #8b5cf6 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* Glass card effect */
    .glass-card {
        background: rgba(26, 26, 36, 0.8);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: var(--accent);
        box-shadow: 0 0 30px var(--accent-glow);
    }
    
    .card-title {
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .card-title::before {
        content: '';
        width: 4px;
        height: 16px;
        background: var(--accent);
        border-radius: 2px;
    }
    
    /* Risk display */
    .risk-display {
        text-align: center;
        padding: 2rem;
    }
    
    .risk-value {
        font-family: 'Inter', sans-serif;
        font-size: 4rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    
    .risk-low { color: var(--success); text-shadow: 0 0 40px rgba(16, 185, 129, 0.5); }
    .risk-medium { color: var(--warning); text-shadow: 0 0 40px rgba(245, 158, 11, 0.5); }
    .risk-high { color: var(--danger); text-shadow: 0 0 40px rgba(239, 68, 68, 0.5); }
    
    .risk-label {
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        flex: 1;
        background: var(--secondary);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        text-align: center;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.25rem;
    }
    
    .metric-value {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    /* Custom Sliders - Clean Dark Style */
    .stSlider {
        padding: 0.25rem 0;
    }
    
    /* Track background */
    .stSlider [data-baseweb="slider"] > div:first-child {
        background: rgb(30, 30, 45) !important;
        height: 8px !important;
        border-radius: 10px !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.4) !important;
    }
    
    /* Filled track */
    .stSlider [data-baseweb="slider"] > div:first-child > div {
        background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
        height: 8px !important;
        border-radius: 10px !important;
    }
    
    /* Thumb */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        width: 20px !important;
        height: 20px !important;
        background: linear-gradient(145deg, #fff, #ddd) !important;
        border: 2px solid #6366f1 !important;
        border-radius: 50% !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3), 0 0 10px rgba(99, 102, 241, 0.3) !important;
    }
    
    .stSlider [data-baseweb="slider"] [role="slider"]:hover {
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.4), 0 0 15px rgba(99, 102, 241, 0.5) !important;
    }
    
    /* Hide the default thumb styling issues */
    .stSlider [data-baseweb="slider"] > div:nth-child(2),
    .stSlider [data-baseweb="slider"] > div:nth-child(3) {
        display: none !important;
    }
    
    /* Input labels */
    .stSlider label, .stSelectbox label, .stNumberInput label {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Number inputs */
    .stNumberInput > div > div > input {
        background: rgb(30, 30, 40) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3) !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: var(--accent) !important;
        box-shadow: 
            inset 0 2px 4px rgba(0, 0, 0, 0.3),
            0 0 10px rgba(99, 102, 241, 0.3) !important;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background: rgb(30, 30, 40) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 8px !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3) !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: var(--accent) !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--secondary);
        border-radius: 12px;
        padding: 0.25rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: var(--text-secondary);
        font-weight: 500;
        padding: 0.75rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--accent) !important;
        color: white !important;
    }
    
    /* DataFrames */
    .stDataFrame {
        border: 1px solid var(--card-border);
        border-radius: 12px;
        overflow: hidden;
    }
    
    .stDataFrame [data-testid="stDataFrameResizable"] {
        background: var(--secondary);
    }
    
    /* Charts */
    .js-plotly-plot .plotly .modebar {
        display: none !important;
    }
    
    /* Section divider */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--card-border), transparent);
        margin: 2rem 0;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .glass-card {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Progress bar for risk */
    .risk-bar-container {
        width: 100%;
        height: 8px;
        background: var(--secondary);
        border-radius: 4px;
        overflow: hidden;
        margin-top: 1rem;
    }
    
    .risk-bar {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    /* Footer */
    .custom-footer {
        text-align: center;
        padding: 2rem;
        color: var(--text-muted);
        font-size: 0.75rem;
    }
    
    .custom-footer a {
        color: var(--accent);
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'risk_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


@st.cache_resource
def get_supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)


@st.cache_data(ttl=60)
def fetch_sample_data():
    """Fetch balanced sample from all years for scatter plots"""
    supabase = get_supabase_client()
    all_data = []
    
    for year in [2021, 2022, 2023, 2024]:
        try:
            response = supabase.table(f'data{year}').select(
                'Year, LTVRatioPercent, Borrower1CreditScoreValue, TotalDebtExpenseRatioPercent, LoanAcquisitionActualUPBAmt, NoteRatePercent, Bank'
            ).limit(2500).execute()
            if response.data:
                all_data.extend(response.data)
        except:
            pass
    
    df = pd.DataFrame(all_data)
    for col in ['LTVRatioPercent', 'TotalDebtExpenseRatioPercent', 'NoteRatePercent', 'LoanAcquisitionActualUPBAmt']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    credit_map = {1: 500, 2: 640, 3: 670, 4: 700, 5: 750, 6: 815, 9: None}
    if 'Borrower1CreditScoreValue' in df.columns:
        df['CreditScore'] = df['Borrower1CreditScoreValue'].map(credit_map)
    
    if 'Bank' in df.columns:
        df['Bank'] = df['Bank'].str.strip()
    
    return df


@st.cache_data(ttl=300)
def fetch_full_aggregates():
    """Fetch aggregated stats from entire database using SQL"""
    supabase = get_supabase_client()
    
    # Overall stats
    overall = supabase.rpc('get_overall_stats').execute()
    
    # Yearly stats
    yearly = supabase.rpc('get_yearly_stats').execute()
    
    # Bank stats
    banks = supabase.rpc('get_bank_stats').execute()
    
    # Risk distribution
    risk = supabase.rpc('get_risk_distribution').execute()
    
    return {
        'overall': overall.data[0] if overall.data else {},
        'yearly': pd.DataFrame(yearly.data) if yearly.data else pd.DataFrame(),
        'banks': pd.DataFrame(banks.data) if banks.data else pd.DataFrame(),
        'risk': pd.DataFrame(risk.data) if risk.data else pd.DataFrame()
    }


@st.cache_data(ttl=300)
def fetch_high_risk_loans(limit=20):
    supabase = get_supabase_client()
    response = supabase.table('all_loans').select(
        'LoanCharacteristicsID, Year, Bank, FIPSStateNumericCode, LTVRatioPercent, Borrower1CreditScoreValue, TotalDebtExpenseRatioPercent, LoanAcquisitionActualUPBAmt, NoteRatePercent'
    ).order('Borrower1CreditScoreValue', desc=False).limit(limit).execute()
    df = pd.DataFrame(response.data)
    df = df[df['Borrower1CreditScoreValue'] > 100]
    return df


def compute_risk_score(ltv, credit_score, dti):
    ltv_score = min(ltv / 100, 1.5)
    credit_norm = 1 - ((credit_score - 300) / 550)
    credit_norm = max(0, min(1, credit_norm))
    dti_score = min(dti / 100, 1)
    risk = (0.4 * ltv_score + 0.35 * credit_norm + 0.25 * dti_score) * 100
    return round(risk, 2)


def predict_risk(model_data, inputs):
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']
    
    input_row = {
        'LTVRatioPercent': inputs.get('ltv_ratio', 80),
        'Borrower1CreditScoreValue': inputs.get('credit_score', 720),
        'TotalDebtExpenseRatioPercent': inputs.get('dti', 35),
        'HousingExpenseRatioPercent': inputs.get('housing_ratio', 28),
        'NoteRatePercent': inputs.get('interest_rate', 6.5),
        'LoanAcquisitionActualUPBAmt': inputs.get('loan_amount', 300000),
        'LoanAmortizationMaxTermMonths': inputs.get('term', 360),
        'TotalMonthlyIncomeAmount': inputs.get('monthly_income', 8000),
        'Borrower1AgeAtApplicationYears': inputs.get('age', 35)
    }
    
    input_df = pd.DataFrame([{feat: input_row.get(feat, 0) for feat in features}])
    input_scaled = scaler.transform(input_df)
    
    # Use decision function for continuous score, then convert to probability-like scale
    decision = model.decision_function(input_scaled)[0]
    # Sigmoid transformation to get 0-100 scale
    prob = 1 / (1 + np.exp(-decision))
    return prob * 100


def sensitivity_analysis(model_data, base_inputs, vary_feature, vary_range):
    results = []
    for value in vary_range:
        inputs = base_inputs.copy()
        inputs[vary_feature] = value
        risk = predict_risk(model_data, inputs)
        results.append({'value': value, 'risk': risk})
    return pd.DataFrame(results)


def get_risk_class(risk_prob):
    if risk_prob < 30:
        return "low", "LOW RISK", "#10b981"
    elif risk_prob < 50:
        return "medium", "MEDIUM RISK", "#f59e0b"
    elif risk_prob < 70:
        return "medium", "HIGH RISK", "#f97316"
    else:
        return "high", "CRITICAL", "#ef4444"


def create_dark_chart(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(18,18,26,0.8)',
        font=dict(family='Inter, sans-serif', color='#a1a1aa'),
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(gridcolor='#2a2a3a', zerolinecolor='#2a2a3a'),
        yaxis=dict(gridcolor='#2a2a3a', zerolinecolor='#2a2a3a'),
    )
    return fig


def main():
    # Header
    st.markdown("""
    <div class="custom-header">
        <div>
            <h1 class="header-title">FHLB Loan Risk Calculator</h1>
            <p class="header-subtitle">Advanced ML-powered risk assessment for mortgage lending</p>
        </div>
        <div class="header-badge">178K+ LOANS ANALYZED</div>
    </div>
    """, unsafe_allow_html=True)
    
    model_data = load_model()
    
    if model_data is None:
        st.error("Model not found. Please run `python model_trainer.py` first.")
        st.stop()
    
    # Main layout
    col_left, col_right = st.columns([1, 2], gap="large")
    
    with col_left:
        st.markdown('<p class="card-title">Loan Parameters</p>', unsafe_allow_html=True)
        
        ltv = st.slider("Loan-to-Value Ratio (%)", 20, 100, 80)
        credit_score = st.slider("Credit Score", 300, 850, 720)
        dti = st.slider("Debt-to-Income (%)", 10, 60, 35)
        housing_ratio = st.slider("Housing Expense Ratio (%)", 10, 50, 28)
        interest_rate = st.slider("Interest Rate (%)", 2.0, 10.0, 6.5, step=0.125)
        loan_amount = st.number_input("Loan Amount ($)", 50000, 2000000, 300000, step=10000)
        term = st.selectbox("Term (months)", [180, 240, 360], index=2)
        monthly_income = st.number_input("Monthly Income ($)", 2000, 50000, 8000, step=500)
        borrower_age = st.slider("Borrower Age", 21, 75, 35)
        
        inputs = {
            'ltv_ratio': ltv, 'credit_score': credit_score, 'dti': dti,
            'housing_ratio': housing_ratio, 'interest_rate': interest_rate,
            'loan_amount': loan_amount, 'term': term,
            'monthly_income': monthly_income, 'age': borrower_age
        }
    
    with col_right:
        risk_prob = predict_risk(model_data, inputs)
        simple_risk = compute_risk_score(ltv, credit_score, dti)
        risk_class, risk_label, risk_color = get_risk_class(risk_prob)
        
        # Risk Assessment Card
        st.markdown('<p class="card-title">Risk Assessment</p>', unsafe_allow_html=True)
        
        # Metrics row
        bar_width = min(risk_prob, 100)
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-label">ML Probability</div>
                <div class="metric-value" style="color: {risk_color}">{risk_prob:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Risk Level</div>
                <div class="metric-value" style="color: {risk_color}">{risk_label}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Simple Score</div>
                <div class="metric-value">{simple_risk:.1f}%</div>
            </div>
        </div>
        <div class="risk-bar-container">
            <div class="risk-bar" style="width: {bar_width}%; background: linear-gradient(90deg, #10b981 0%, #f59e0b 50%, #ef4444 100%);"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Sensitivity Analysis Card
        st.markdown('<p class="card-title" style="margin-top: 1.5rem;">Sensitivity Analysis</p>', unsafe_allow_html=True)
        
        sens_feature = st.selectbox(
            "Analyze impact of:",
            ['ltv_ratio', 'credit_score', 'dti', 'interest_rate'],
            format_func=lambda x: {'ltv_ratio': 'LTV Ratio', 'credit_score': 'Credit Score', 
                                   'dti': 'Debt-to-Income', 'interest_rate': 'Interest Rate'}.get(x, x)
        )
        
        if sens_feature == 'ltv_ratio':
            vary_range, x_label = np.arange(50, 101, 5), "LTV Ratio (%)"
        elif sens_feature == 'credit_score':
            vary_range, x_label = np.arange(500, 851, 25), "Credit Score"
        elif sens_feature == 'dti':
            vary_range, x_label = np.arange(20, 61, 5), "Debt-to-Income (%)"
        else:
            vary_range, x_label = np.arange(3, 10.5, 0.5), "Interest Rate (%)"
        
        sens_df = sensitivity_analysis(model_data, inputs, sens_feature, vary_range)
        
        fig_sens = px.area(sens_df, x='value', y='risk', 
                          labels={'value': x_label, 'risk': 'Risk Probability (%)'})
        fig_sens.update_traces(fill='tozeroy', line=dict(color='#6366f1', width=3),
                              fillcolor='rgba(99, 102, 241, 0.2)')
        fig_sens.add_hline(y=risk_prob, line_dash="dash", line_color="#ef4444",
                          annotation_text="Current", annotation_position="right")
        fig_sens = create_dark_chart(fig_sens)
        fig_sens.update_layout(height=280, showlegend=False)
        st.plotly_chart(fig_sens, width="stretch")
    
    # Divider
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Database Insights
    st.markdown('<p class="card-title">Database Insights <span style="font-size: 0.7em; color: #a855f7;">(Full Database: 178K+ Loans)</span></p>', unsafe_allow_html=True)
    
    try:
        # Fetch full aggregates from database
        agg_data = fetch_full_aggregates()
        sample_df = fetch_sample_data()
        
        overall = agg_data['overall']
        yearly_df = agg_data['yearly']
        banks_df = agg_data['banks']
        risk_df = agg_data['risk']
        
        # Key metrics row - from FULL database
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-label">Total Loans</div>
                <div class="metric-value">{overall.get('total_loans', 0):,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg LTV</div>
                <div class="metric-value">{overall.get('avg_ltv', 0):.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg DTI</div>
                <div class="metric-value">{overall.get('avg_dti', 0):.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Interest Rate</div>
                <div class="metric-value">{overall.get('avg_rate', 0):.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Volume</div>
                <div class="metric-value">${overall.get('total_volume', 0)/1e9:.1f}B</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["Year Trends", "Risk Analysis", "Distributions", "Top Lenders"])
        
        with tab1:
            # Yearly trends from FULL database
            if len(yearly_df) > 0:
                c1, c2 = st.columns(2)
                with c1:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=yearly_df['year'], y=yearly_df['loan_count'], name='Loan Volume',
                                        marker_color='#6366f1', opacity=0.7))
                    fig.add_trace(go.Scatter(x=yearly_df['year'], y=yearly_df['avg_rate']*10000, name='Interest Rate',
                                            mode='lines+markers', line=dict(color='#ef4444', width=3),
                                            yaxis='y2'))
                    fig.update_layout(
                        title='Loan Volume vs Interest Rate Trend',
                        yaxis=dict(title='Number of Loans'),
                        yaxis2=dict(title='Interest Rate (%)', overlaying='y', side='right', 
                                   tickformat='.2f', tickvals=[20000,40000,60000], ticktext=['2%','4%','6%']),
                        height=350, showlegend=True, legend=dict(x=0.02, y=0.98)
                    )
                    fig = create_dark_chart(fig)
                    st.plotly_chart(fig, width="stretch")
                
                with c2:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=yearly_df['year'], y=yearly_df['avg_ltv'], name='LTV %',
                                            mode='lines+markers', line=dict(color='#6366f1', width=3)))
                    fig.add_trace(go.Scatter(x=yearly_df['year'], y=yearly_df['avg_dti'], name='DTI %',
                                            mode='lines+markers', line=dict(color='#10b981', width=3)))
                    fig.update_layout(title='LTV vs DTI Trends', yaxis_title='Percentage', height=350)
                    fig = create_dark_chart(fig)
                    st.plotly_chart(fig, width="stretch")
                
                # Loan volume trend
                fig = go.Figure()
                fig.add_trace(go.Bar(x=yearly_df['year'], y=yearly_df['total_volume']/1e9, name='Total Volume ($B)',
                                    marker_color='#6366f1'))
                fig.update_layout(title='Total Loan Volume by Year', height=300, yaxis_title='Volume (Billions $)')
                fig = create_dark_chart(fig)
                st.plotly_chart(fig, width="stretch")
        
        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                # Scatter plot uses SAMPLE data
                credit_df = sample_df[sample_df['CreditScore'].notna()]
                sample = credit_df.sample(min(2000, len(credit_df))) if len(credit_df) > 0 else credit_df
                if len(sample) > 0:
                    fig = px.scatter(sample, x='LTVRatioPercent', y='CreditScore',
                                    color='TotalDebtExpenseRatioPercent', 
                                    color_continuous_scale='RdYlGn_r',
                                    opacity=0.6,
                                    title='Credit Score vs LTV (10K sample)')
                    fig.update_layout(height=400, xaxis_title='LTV Ratio (%)', yaxis_title='Credit Score')
                    fig = create_dark_chart(fig)
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("No credit score data available")
            
            with c2:
                # Risk distribution from FULL database
                if len(risk_df) > 0:
                    fig = px.pie(risk_df, values='loan_count', names='risk_category',
                                color='risk_category',
                                color_discrete_map={'Low Risk': '#10b981', 'Medium Risk': '#f59e0b', 'High Risk': '#ef4444'},
                                title='Risk Distribution (Full Database)', hole=0.4)
                    fig.update_traces(textinfo='percent+label')
                    fig.update_layout(height=400)
                    fig = create_dark_chart(fig)
                    st.plotly_chart(fig, width="stretch")
        
        with tab3:
            # Distributions use SAMPLE data for histograms
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(sample_df, x='LTVRatioPercent', nbins=40, 
                                  color_discrete_sequence=['#6366f1'],
                                  title='LTV Ratio Distribution')
                fig.update_layout(height=320, xaxis_title='LTV (%)', yaxis_title='Count')
                fig = create_dark_chart(fig)
                st.plotly_chart(fig, width="stretch")
                
                fig = px.histogram(sample_df, x='TotalDebtExpenseRatioPercent', nbins=40,
                                  color_discrete_sequence=['#f59e0b'],
                                  title='Debt-to-Income Distribution')
                fig.update_layout(height=320, xaxis_title='DTI (%)', yaxis_title='Count')
                fig = create_dark_chart(fig)
                st.plotly_chart(fig, width="stretch")
            
            with c2:
                credit_df = sample_df[sample_df['CreditScore'].notna()]
                fig = px.histogram(credit_df, x='CreditScore', nbins=20,
                                  color_discrete_sequence=['#10b981'],
                                  title='Credit Score Distribution')
                fig.update_layout(height=320, xaxis_title='Credit Score', yaxis_title='Count')
                fig = create_dark_chart(fig)
                st.plotly_chart(fig, width="stretch")
                
                fig = px.histogram(sample_df, x='NoteRatePercent', nbins=40,
                                  color_discrete_sequence=['#ef4444'],
                                  title='Interest Rate Distribution')
                fig.update_layout(height=320, xaxis_title='Interest Rate (%)', yaxis_title='Count')
                fig = create_dark_chart(fig)
                st.plotly_chart(fig, width="stretch")
        
        with tab4:
            # Top lenders from FULL database
            if len(banks_df) > 0:
                top_banks = banks_df.head(10)
                
                fig = px.bar(top_banks, x='bank', y='loan_count',
                            color='avg_rate', color_continuous_scale='RdYlGn_r',
                            title='Top 10 FHLB Banks by Loan Volume (Full Database)')
                fig.update_layout(height=400, xaxis_tickangle=-45, xaxis_title='FHLB Bank', yaxis_title='Number of Loans')
                fig = create_dark_chart(fig)
                st.plotly_chart(fig, width="stretch")
                
                # Show table with full stats
                display_df = top_banks.copy()
                display_df.columns = ['Bank', 'Loan Count', 'Avg LTV', 'Avg DTI', 'Avg Rate', 'Avg Loan', 'Total Volume']
                display_df['Loan Count'] = display_df['Loan Count'].apply(lambda x: f"{x:,}")
                display_df['Total Volume'] = display_df['Total Volume'].apply(lambda x: f"${x/1e9:.2f}B" if x > 1e9 else f"${x/1e6:.0f}M")
                display_df['Avg Loan'] = display_df['Avg Loan'].apply(lambda x: f"${x/1e3:.0f}K")
                display_df['Avg Rate'] = display_df['Avg Rate'].apply(lambda x: f"{x:.2f}%")
                display_df['Avg LTV'] = display_df['Avg LTV'].apply(lambda x: f"{x:.1f}%")
                display_df['Avg DTI'] = display_df['Avg DTI'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                st.caption(f"Showing all {len(banks_df)} FHLB banks from the complete database of {overall.get('total_loans', 0):,} loans")
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        import traceback
        st.code(traceback.format_exc())
    
    # Footer
    st.markdown("""
    <div class="custom-footer">
        <p>Data: FHLB Public Use Database (2021-2024) · 178,162 loans · Model: Logistic Regression</p>
        <p style="margin-top: 0.5rem; opacity: 0.7;">For demonstration purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
