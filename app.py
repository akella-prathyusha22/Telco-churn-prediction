import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="centered"
)

# ── Load model and scaler ─────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# ── Header ────────────────────────────────────────────────────
st.title("Telco Customer Churn Predictor")
st.markdown("Enter customer details below to predict churn risk.")
st.divider()

# ── Input form ────────────────────────────────────────────────
st.subheader("Customer Details")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 20, 120, 65)
    total_charges = st.slider("Total Charges ($)", 0, 9000, 1000)
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])

with col2:
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
    internet_service = st.selectbox("Internet Service", 
                                     ["DSL", "Fiber Optic", "None"])
    online_security = st.selectbox("Online Security", ["No", "Yes"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])

col3, col4 = st.columns(2)

with col3:
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
    contract = st.selectbox("Contract Type", 
                             ["Month-to-Month", "One Year", "Two Year"])

with col4:
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", 
                                   ["Bank Transfer", "Credit Card", 
                                    "Electronic Check", "Mailed Check"])

st.divider()

# ── Feature engineering ───────────────────────────────────────
def prepare_input():
    # Binary encoding
    def yn(val): return 1 if val == "Yes" else 0

    # Engineered features
    charges_per_tenure = monthly_charges / (tenure + 1)
    is_long_term = 1 if contract in ["One Year", "Two Year"] else 0

    # Internet service
    internet_dsl = 1 if internet_service == "DSL" else 0
    internet_fiber = 1 if internet_service == "Fiber Optic" else 0
    internet_none = 1 if internet_service == "None" else 0

    # Contract
    contract_monthly = 1 if contract == "Month-to-Month" else 0
    contract_one = 1 if contract == "One Year" else 0
    contract_two = 1 if contract == "Two Year" else 0

    # Payment
    pay_bank = 1 if payment_method == "Bank Transfer" else 0
    pay_card = 1 if payment_method == "Credit Card" else 0
    pay_echeck = 1 if payment_method == "Electronic Check" else 0
    pay_mail = 1 if payment_method == "Mailed Check" else 0

    features = {
        'gender': 0,  # not collected — neutral default
        'SeniorCitizen': yn(senior_citizen),
        'Partner': yn(partner),
        'Dependents': yn(dependents),
        'tenure': tenure,
        'PhoneService': yn(phone_service),
        'MultipleLines': yn(multiple_lines),
        'OnlineSecurity': yn(online_security),
        'OnlineBackup': yn(online_backup),
        'DeviceProtection': yn(device_protection),
        'TechSupport': yn(tech_support),
        'StreamingTV': yn(streaming_tv),
        'StreamingMovies': yn(streaming_movies),
        'PaperlessBilling': yn(paperless_billing),
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'payment_bank_transfer': pay_bank,
        'payment_credit_card': pay_card,
        'payment_electronic_check': pay_echeck,
        'payment_mailed_check': pay_mail,
        'contract_monthly': contract_monthly,
        'contract_one_year': contract_one,
        'contract_two_year': contract_two,
        'internet_service_dsl': internet_dsl,
        'internet_service_fiber': internet_fiber,
        'internet_service_none': internet_none,
        'charges_per_tenure': charges_per_tenure,
        'is_long_term_contract': is_long_term
    }

    return pd.DataFrame([features])

# ── Predict button ────────────────────────────────────────────
if st.button("Predict Churn Risk", type="primary", use_container_width=True):

    input_df = prepare_input()
    input_scaled = scaler.transform(input_df)
    churn_prob = model.predict_proba(input_scaled)[0][1]
    churn_pred = 1 if churn_prob >= 0.4 else 0

    st.divider()
    st.subheader("Prediction Result")

    # Risk gauge
    if churn_prob >= 0.7:
        st.error(f"🔴 HIGH RISK — {churn_prob*100:.1f}% churn probability")
        recommendation = "Immediate action required. Offer a contract upgrade with a 20% loyalty discount."
    elif churn_prob >= 0.4:
        st.warning(f"🟡 MEDIUM RISK — {churn_prob*100:.1f}% churn probability")
        recommendation = "Schedule a proactive check-in call. Consider a contract conversion offer."
    else:
        st.success(f"🟢 LOW RISK — {churn_prob*100:.1f}% churn probability")
        recommendation = "Customer is stable. Continue standard engagement."

    # Key risk factors
    st.subheader("Key Risk Factors")
    factors = []
    if contract == "Month-to-Month":
        factors.append("⚠️ No long-term commitment — highest churn risk contract type")
    if tenure < 12:
        factors.append("⚠️ Early-life customer — 80% of churners leave in first 12 months")
    if internet_service == "Fiber Optic":
        factors.append("⚠️ Fiber Optic subscriber — premium price, higher churn segment")
    if payment_method == "Electronic Check":
        factors.append("⚠️ Electronic check payment — correlated with higher churn")
    if monthly_charges > 70:
        factors.append("⚠️ High monthly charges — price sensitivity risk")

    if factors:
        for f in factors:
            st.markdown(f)
    else:
        st.markdown("✅ No major risk factors detected")

    # Recommendation
    st.subheader("Recommended Action")
    st.info(f"💡 {recommendation}")
