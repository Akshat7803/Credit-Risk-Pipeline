# Credit Risk Scoring & Risk Segmentation Pipeline

A two-stage ML pipeline for credit risk assessment on 51,000+ loan applicants.

## Pipeline
- **Stage 1:** XGBoost regression — predicts credit score (R²=0.89, RMSE=6.8)
- **Stage 2:** XGBoost classifier — segments applicants into P1/P2/P3/P4 risk tiers (F1-macro=0.70)

## Features
- SHAP explainability for risk factor attribution
- Streamlit app with real-time risk prediction
- LLM integration (Llama 3.3 via Groq) for plain-English decision explanations
- What-if scenario analysis for loan officers

## Setup
pip install streamlit groq xgboost pandas numpy
streamlit run credit_risk_explainer.py

## Tech Stack
Python · XGBoost · SHAP · Streamlit · Groq API
