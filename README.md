# Telco Customer Churn Prediction

## Project Overview
A complete end-to-end data science project predicting customer churn for a telecom company using machine learning. The goal is to identify at-risk customers before they leave, enabling proactive retention strategies.

## Business Problem
Customer churn costs telecom companies millions annually. Acquiring a new customer costs 5x more than retaining an existing one. This project builds a model that flags high-risk customers so retention teams can intervene early.

## Dataset
- **Source:** IBM Telco Customer Churn Dataset (via Kaggle)
- **Size:** 7,032 customers, 21 features
- **Target:** Binary churn label (26.6% churn rate)

## Project Structure

-telco-churn-prediction/telco_churn_analysis.ipynb   # Main notebook - full analysis
-telco-churn-prediction/telco_churn_clean.csv        # Cleaned dataset
-README.md


## Key Findings
- **Contract type** is the strongest churn predictor - month-to-month customers churn at 4x the rate of long-term contract holders
- **Churn is an early-life problem** — 80% of churners leave within the first 10 months
- **Fiber Optic customers** are the highest risk, premium pricing combined with no long-term commitment
- **Engineered feature** 'charges_per_tenure' ranked in the top 15 predictors, validating the feature engineering approach

## Technical Approach
| Step | Details |
|---|---|
| Data Cleaning | Fixed TotalCharges dtype, dropped 11 zero-tenure rows |
| Feature Engineering | Created charges_per_tenure, is_long_term_contract |
| Encoding | Binary mapping + get_dummies for categorical columns |
| Models | Logistic Regression, Random Forest |
| Evaluation | Classification report, ROC-AUC, threshold tuning |

## Results
| Model | Accuracy | Churn Recall | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 80% | 55% | 0.83 |
| Logistic Regression (Tuned) | 79% | 68% | 0.83 |
| Random Forest | 78% | 48% | 0.81 |
| Random Forest (Tuned) | 77% | 47% | 0.81 |

**Final model:** Logistic Regression with threshold=0.4
- Best Recall (68%) for catching at-risk customers
- ROC-AUC of 0.83 — significantly above random baseline

## Business Recommendations
1. **90-day onboarding program**: proactive check-ins in first 3 months
2. **Contract conversion incentives**: discount for switching from month-to-month to one-year within the first 6 months
3. **Fiber Optic loyalty program** — targeted retention for high-charge, short-tenure Fiber customers
4. **Monthly churn scoring** — score all active customers and prioritize retention calls for top 20% highest risk

## Tools & Libraries
![Python](https://img.shields.io/badge/Python-3.9-blue)
![Pandas](https://img.shields.io/badge/Pandas-1.3-green)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.0-orange)
![Seaborn](https://img.shields.io/badge/Seaborn-0.11-purple)

## Next Steps
- Test XGBoost and LightGBM for Recall improvement
- Build a Streamlit app for real-time churn scoring
- Collect additional features: customer service logs, network quality scores

## Author
Prathyusha Akella — Data Scientist  | https://www.linkedin.com/in/prathyusha-akella
