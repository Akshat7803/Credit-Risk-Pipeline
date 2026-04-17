# Credit Risk Scoring & Risk Segmentation Pipeline
 
A two-stage machine learning pipeline for credit risk assessment, built on 51,000+ real loan applicants. The system predicts credit scores, segments borrowers into risk tiers, and generates plain-English explanations for loan officers using an LLM — mirroring how credit bureaus and lending institutions operate in practice.
 
---
 
## Problem Statement
 
Lending institutions need to assess creditworthiness quickly and consistently across thousands of applicants. Manual assessment is slow, inconsistent, and hard to justify to regulators. This project builds an automated, explainable credit risk pipeline that:
- Predicts an applicant's credit score
- Classifies them into a risk tier for lending decisions
- Explains the decision in plain English for loan officers
---
 
## Pipeline Architecture
 
```
Raw Applicant Data
       │
       ▼
┌─────────────────────────────┐
│  Stage 1 — Regression       │
│  XGBoost Credit Score Model │
│  R² = 0.894  │  RMSE = 6.46 │
└─────────────┬───────────────┘
              │ Predicted Credit Score
              ▼
┌─────────────────────────────────────────┐
│  Stage 2 — Classification               │
│  XGBoost Risk Tier Segmentation         │
│  AUC-ROC = 0.937  │  Gini = 0.874       │
│                                         │
│  P1 – Prime       → Approve freely      │
│  P2 – Near Prime  → Approve             │
│  P3 – Sub Prime   → Approve with caution│
│  P4 – High Risk   → Decline             │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│  LLM Explainer              │
│  Llama 3.3 70B via Groq     │
│  Plain-English explanation  │
│  for loan officers          │
└─────────────────────────────┘
```
 
---
 
## Key Features
 
- **Two-stage pipeline** — regression feeds into classification, mirroring real credit bureau architecture
- **SHAP explainability** — TreeExplainer identifies top risk drivers per applicant and per risk tier
- **Data leakage prevention** — Stage 2 uses predicted credit scores (not actual), ensuring valid out-of-sample evaluation
- **Smart missing value handling** — missingness indicator flags for delinquency and utilization features based on domain logic
- **Multicollinearity reduction** — VIF and correlation filtering reduced features from ~80 to 55
- **Class imbalance handling** — balanced class weights and sample weighting across all classifiers
- **Interactive Streamlit app** — real-time risk prediction with what-if scenario analysis
- **LLM integration** — Groq API (Llama 3.3 70B) generates regulatory-ready plain-English explanations
---
 
## Model Performance
 
### Stage 1 — Credit Score Regression
 
| Model | R² | RMSE | MAE |
|---|---|---|---|
| Linear Regression (baseline) | 0.897 | 6.46 | 5.47 |
| XGBoost (default) | 0.894 | 6.57 | 5.53 |
| XGBoost (tuned) | 0.894 | 6.56 | 5.51 |
 
### Stage 2 — Risk Tier Classification
 
| Model | Accuracy | F1-macro | AUC-ROC | Gini | CV F1 (5-fold) |
|---|---|---|---|---|---|
| Logistic Regression | 0.742 | 0.715 | 0.921 | 0.843 | 0.719 ± 0.003 |
| Random Forest | 0.793 | 0.690 | 0.937 | 0.874 | 0.695 ± 0.003 |
| XGBoost | 0.742 | 0.711 | 0.937 | 0.874 | 0.717 ± 0.003 |
 
### Per-class Performance — Best Model (Random Forest)
 
| Risk Tier | Precision | Recall | F1 |
|---|---|---|---|
| P1 – Prime | 0.80 | 0.80 | 0.80 |
| P2 – Near Prime | 0.83 | 0.93 | 0.88 |
| P3 – Sub Prime | 0.46 | 0.25 | 0.33 |
| P4 – High Risk | 0.77 | 0.74 | 0.76 |

---
## App Screenshots
<img width="1920" height="1080" alt="Screenshot (845)" src="https://github.com/user-attachments/assets/5a10919f-a758-486e-9d55-c79e8a8cab40" />

 
## Tech Stack
 
| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| ML Models | XGBoost, Scikit-learn |
| Explainability | SHAP |
| App Framework | Streamlit |
| LLM | Llama 3.3 70B via Groq API |
| Data | Pandas, NumPy |
 
---
 
## Author
 
**Your Name**
[LinkedIn](https://www.linkedin.com/in/akshatjain71/) · [GitHub](https://github.com/Akshat7803/)
