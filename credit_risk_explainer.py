"""
Credit Risk Analyzer with LLM Explainer
========================================
Setup:
    pip install streamlit groq xgboost pandas numpy

Place these files in the same folder:
    - xgb_clf.json
    - scaler_params.json
    - le_params.json

Run:
    streamlit run credit_risk_explainer.py
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from groq import Groq
from xgboost import XGBClassifier

st.set_page_config(page_title="Credit Risk Analyzer", page_icon="🏦", layout="wide")

st.markdown("""
<style>
    .risk-badge { display:inline-block; padding:6px 18px; border-radius:20px; font-weight:600; font-size:15px; }
    .p1 { background:#d1fae5; color:#065f46; }
    .p2 { background:#dbeafe; color:#1e40af; }
    .p3 { background:#fef3c7; color:#92400e; }
    .p4 { background:#fee2e2; color:#991b1b; }
    .metric-box { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:14px 18px; text-align:center; }
    .metric-label { font-size:12px; color:#64748b; margin-bottom:4px; }
    .metric-value { font-size:22px; font-weight:700; color:#1e293b; }
    .explanation-box { background:#f0f9ff; border-left:4px solid #0ea5e9; border-radius:0 10px 10px 0; padding:18px 20px; font-size:15px; line-height:1.7; color:#0f172a; }
</style>
""", unsafe_allow_html=True)

RISK_TIERS = {
    0: {"label": "P1 – Prime",      "badge": "p1"},
    1: {"label": "P2 – Near Prime", "badge": "p2"},
    2: {"label": "P3 – Sub Prime",  "badge": "p3"},
    3: {"label": "P4 – High Risk",  "badge": "p4"},
}

FEATURE_MEDIANS = {
    'Total_TL': 2.0, 'Tot_Active_TL': 1.0, 'Total_TL_opened_L6M': 0.0,
    'Tot_TL_closed_L6M': 0.0, 'pct_tl_open_L6M': 0.0, 'pct_tl_closed_L6M': 0.0,
    'pct_active_tl': 0.556, 'pct_tl_open_L12M': 0.333, 'pct_tl_closed_L12M': 0.0,
    'Tot_Missed_Pmnt': 0.0, 'Auto_TL': 0.0, 'CC_TL': 0.0, 'Consumer_TL': 0.0,
    'Home_TL': 0.0, 'PL_TL': 0.0, 'Other_TL': 0.0, 'Age_Oldest_TL': 33.0,
    'Age_Newest_TL': 8.0, 'time_since_recent_payment': 74.0,
    'time_since_recent_deliquency': -1.0, 'num_times_delinquent': 0.0,
    'max_delinquency_level': 0.0, 'num_deliq_6mts': 0.0, 'max_deliq_6mts': 0.0,
    'max_deliq_12mts': 0.0, 'num_times_60p_dpd': 0.0, 'num_std': 0.0,
    'num_std_6mts': 0.0, 'num_sub': 0.0, 'num_sub_6mts': 0.0, 'num_sub_12mts': 0.0,
    'num_dbt': 0.0, 'num_dbt_6mts': 0.0, 'num_lss': 0.0, 'num_lss_6mts': 0.0,
    'tot_enq': 3.0, 'CC_enq': 0.0, 'CC_enq_L6m': 0.0, 'PL_enq_L6m': 0.0,
    'time_since_recent_enq': 45.0, 'enq_L3m': 0.0, 'MARITALSTATUS': 0.0,
    'EDUCATION': 1.0, 'AGE': 32.0, 'GENDER': 1.0, 'NETMONTHLYINCOME': 23000.0,
    'Time_With_Curr_Empr': 94.0, 'pct_currentBal_all_TL': 0.619,
    'CC_utilization': 0.0, 'PL_utilization': 0.0, 'pct_PL_enq_L6m_of_ever': 0.0,
    'pct_CC_enq_L6m_of_ever': 0.0, 'max_unsec_exposure_inPct': 0.335,
    'HL_Flag': 0.0, 'GL_Flag': 0.0, 'CC_utilization_missing': 1.0,
    'PL_utilization_missing': 1.0, 'max_delinquency_level_missing': 1.0,
    'max_unsec_exposure_missing': 0.0, 'max_deliq_6mts_missing': 0.0,
    'last_prod_enq2_CC': 0.0, 'last_prod_enq2_ConsumerLoan': 0.0,
    'last_prod_enq2_HL': 0.0, 'last_prod_enq2_PL': 0.0,
    'last_prod_enq2_others': 0.0, 'first_prod_enq2_CC': 0.0,
    'first_prod_enq2_ConsumerLoan': 0.0, 'first_prod_enq2_HL': 0.0,
    'first_prod_enq2_PL': 0.0, 'first_prod_enq2_others': 1.0,
    'predicted_credit_score': 679.83,
}

class SimpleScaler:
    def __init__(self, mean, scale):
        self.mean_  = np.array(mean)
        self.scale_ = np.array(scale)

    def transform(self, X):
        return (np.array(X, dtype=float) - self.mean_) / self.scale_

class SimpleLabelEncoder:
    def __init__(self, classes):
        self.classes_ = np.array(classes)

    def inverse_transform(self, X):
        return self.classes_[np.array(X)]

@st.cache_resource
def load_models():
    clf = XGBClassifier()
    clf.load_model("xgb_clf.json")

    with open("scaler_params.json") as f:
        sp = json.load(f)
    scaler = SimpleScaler(sp["mean"], sp["scale"])

    with open("le_params.json") as f:
        lp = json.load(f)
    le = SimpleLabelEncoder(lp["classes"])

    # Global feature importance from XGBoost (no SHAP needed)
    importance = clf.get_booster().get_score(importance_type='gain')
    return clf, scaler, le, importance

try:
    xgb_clf, scaler_cla, le, global_importance = load_models()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    MODEL_ERROR  = str(e)

def predict(input_df):
    X_scaled   = scaler_cla.transform(input_df.values)
    cls        = int(xgb_clf.predict(X_scaled)[0])
    proba      = xgb_clf.predict_proba(X_scaled)[0]
    confidence = round(float(proba.max()), 2)

    # Feature contribution: importance * deviation from median
    feature_names = input_df.columns.tolist()
    contributions = {}
    for feat in feature_names:
        if feat in global_importance:
            median_val = FEATURE_MEDIANS.get(feat, 0)
            deviation  = float(input_df[feat].iloc[0]) - median_val
            contributions[feat] = global_importance[feat] * deviation / 1000

    top = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:8])
    return cls, confidence, top

def get_llm_explanation(sidebar_vals, risk_tier, contributions, api_key, what_if=None):
    client = Groq(api_key=api_key)

    factor_lines = "\n".join(
        f"  - {k}: {'INCREASES' if v > 0 else 'REDUCES'} risk"
        for k, v in sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:4]
    )
    what_if_section = f"\n\nLoan officer asks: {what_if}\nAnswer this directly at the end." if what_if else ""

    prompt = f"""You are an expert credit risk analyst at an Indian bank.

Applicant profile:
- Monthly income: ₹{sidebar_vals['monthly_income']:,}
- Loan requested: ₹{sidebar_vals['loan_amount']:,}
- Age: {sidebar_vals['age']} years
- Employment tenure: {sidebar_vals['employment_tenure']} months
- Credit utilization: {sidebar_vals['credit_utilization']}%
- Delinquencies (ever): {sidebar_vals['num_delinquencies']}
- Missed payments: {sidebar_vals['on_time_pmt_pct']}
- Total enquiries: {sidebar_vals['recent_inquiries']}
- Total trade lines: {sidebar_vals['total_tl']}
- Age of oldest credit line: {sidebar_vals['credit_age']} months

Model decision: {risk_tier}

Key risk factors driving this decision:
{factor_lines}

Write a 3-paragraph explanation for the loan officer:
1. Overall risk assessment and primary reason for this classification.
2. Top 2-3 contributing factors in plain language with actual values.
3. Specific, actionable recommendation — approval conditions or improvement steps.

Be direct, cite actual numbers. No bullet points — flowing paragraphs only.{what_if_section}"""

    msg = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.choices[0].message.content

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏦 Credit Risk Analyzer")
    st.caption("XGBoost + Groq AI · Real Model")

    if not MODEL_LOADED:
        st.error(f"Model load failed: {MODEL_ERROR}")
        st.stop()
    else:
        st.success("Model loaded ✓")

    api_key = st.text_input("Groq API Key", type="password",
                             value=os.environ.get("ANTHROPIC_API_KEY", ""),
                             placeholder="gsk_...")
    st.divider()
    st.subheader("Applicant Profile")

    monthly_income       = st.number_input("Monthly Income (₹)", 10000, 500000, 23000, 1000)
    loan_amount          = st.number_input("Loan Amount Requested (₹)", 50000, 5000000, 300000, 10000)
    age                  = st.slider("Age (years)", 21, 65, 32)
    employment_tenure    = st.slider("Employment Tenure (months)", 0, 300, 94)
    credit_age           = st.slider("Age of Oldest Credit Line (months)", 6, 300, 33)
    credit_utilization   = st.slider("Credit Utilization (%)", 0, 100, 0)
    num_delinquencies    = st.slider("Number of Delinquencies (ever)", 0, 10, 0)
    on_time_pmt_pct      = st.slider("Missed Payments (count)", 0, 20, 0)
    recent_inquiries     = st.slider("Total Enquiries", 0, 20, 3)
    total_tl             = st.slider("Total Trade Lines", 0, 20, 2)
    predicted_credit_score = st.slider("Predicted Credit Score", 300, 900, 680)

    analyze_btn = st.button("Analyze Application", type="primary", use_container_width=True)

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("Loan Application Assessment")

if not analyze_btn:
    st.info("Configure the applicant profile in the sidebar and click **Analyze Application**.")
    st.stop()

input_row = FEATURE_MEDIANS.copy()
input_row['AGE']                           = age
input_row['NETMONTHLYINCOME']              = monthly_income
input_row['Time_With_Curr_Empr']           = employment_tenure
input_row['Age_Oldest_TL']                 = credit_age
input_row['CC_utilization']                = credit_utilization / 100
input_row['max_delinquency_level']         = num_delinquencies*100
input_row['num_times_delinquent']          = num_delinquencies
input_row['Tot_Missed_Pmnt']               = on_time_pmt_pct
input_row['tot_enq']                       = recent_inquiries
input_row['Total_TL']                      = total_tl
input_row['CC_utilization_missing']        = 0 if credit_utilization > 0 else 1
input_row['max_delinquency_level_missing'] = 0 if num_delinquencies > 0 else 1
input_row['predicted_credit_score'] = predicted_credit_score

feature_names = xgb_clf.get_booster().feature_names or list(FEATURE_MEDIANS.keys())
input_df = pd.DataFrame([input_row])[feature_names]

sidebar_vals = {
    "monthly_income": monthly_income, "loan_amount": loan_amount,
    "age": age, "employment_tenure": employment_tenure,
    "credit_age": credit_age, "credit_utilization": credit_utilization,
    "num_delinquencies": num_delinquencies, "on_time_pmt_pct": on_time_pmt_pct,
    "recent_inquiries": recent_inquiries, "total_tl": total_tl,
    "predicted_credit_score": predicted_credit_score,
}

with st.spinner("Running model..."):
    cls, confidence, contributions = predict(input_df)

tier_info = RISK_TIERS[cls]

# ── Metric cards ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-box"><div class="metric-label">Risk Tier</div><div style="margin-top:6px"><span class="risk-badge {tier_info["badge"]}">{tier_info["label"]}</span></div></div>', unsafe_allow_html=True)
with c2:
    conf_display = confidence if confidence <= 1 else confidence / 100
    st.markdown(f'<div class="metric-box"><div class="metric-label">Confidence</div><div class="metric-value">{conf_display*100:.0f}%</div></div>', unsafe_allow_html=True)
with c3:
    lti = round(loan_amount / (monthly_income * 12), 2)
    st.markdown(f'<div class="metric-box"><div class="metric-label">Loan-to-Income</div><div class="metric-value">{lti}x</div></div>', unsafe_allow_html=True)
with c4:
    emi = round(loan_amount * 0.02)
    st.markdown(f'<div class="metric-box"><div class="metric-label">Est. EMI Burden</div><div class="metric-value">₹{emi:,}/mo</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Risk Factor Analysis")
    st.caption("Top 8 features for this applicant. Red = increases risk · Blue = reduces risk")
    max_abs = max(abs(v) for v in contributions.values()) or 1
    for feat, val in sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True):
        pct   = abs(val) / max_abs * 100
        color = "#ef4444" if val > 0 else "#3b82f6"
        arrow = "▲" if val > 0 else "▼"
        st.markdown(f"""
        <div style="margin-bottom:12px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span style="font-size:13px;">{feat}</span>
                <span style="font-size:13px;color:{color};font-weight:600;">{arrow}</span>
            </div>
            <div style="background:#e2e8f0;border-radius:4px;height:10px;">
                <div style="background:{color};width:{pct:.0f}%;height:10px;border-radius:4px;"></div>
            </div>
        </div>""", unsafe_allow_html=True)

with right:
    st.subheader("AI-Generated Explanation")
    if not api_key:
        st.warning("Enter your Groq API key in the sidebar.")
    else:
        with st.spinner("Generating the explanation..."):
            try:
                expl = get_llm_explanation(sidebar_vals, tier_info["label"], contributions, api_key)
                st.markdown(f'<div class="explanation-box">{expl}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"API error: {e}")

st.divider()
st.subheader("💬 Ask the AI Analyst")
st.caption("E.g. 'What if they had no delinquencies?' or 'What would change this to P2?'")

q = st.text_input("Your question", placeholder="What would this applicant need to improve to qualify for P2?")
if q and api_key:
    with st.spinner("Thinking..."):
        try:
            ans = get_llm_explanation(sidebar_vals, tier_info["label"], contributions, api_key, what_if=q)
            st.markdown(f'<div class="explanation-box">{ans}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"API error: {e}")
elif q and not api_key:
    st.warning("Enter your Groq API key in the sidebar.")
