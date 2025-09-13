import streamlit as st
import pandas as pd
import joblib
import os
# =======================
# Load models and columns
# =======================


# build the path relative to the app.py location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

lgb_model = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'lgb_model.pkl'))
gam_model = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'gam_model.pkl'))
X_columns = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'X_columns.pkl'))

# Categorical options for dropdowns
categorical_features = {
    "Gender": ["m", "f"],
    "Married": ["y", "n"],
    "BankCustomer": ["y", "n"],
    "Industry": ["Industrials", "Finance", "Technology", "Healthcare"],
    "Ethnicity": ["White", "Black", "Asian", "Hispanic", "Other"],
    "PriorDefault": ["y", "n"],
    "Employed": ["y", "n"],
    "Citizen": ["By Birth", "By other means"],
    "IncomeType": ["part_time/freelancer", "NA", "full_time", "full_time/freelancer"],
    "AmbiguousIncome": ["y", "n"],   # NA removed
    "IncomeRate": ["Monthly", "Yearly", "NA"]   # NA added
}

# =======================
# Page Config
# =======================
st.set_page_config(page_title="Credit Card Approval Predictor", page_icon="💳", layout="wide")

# =======================
# Custom CSS
# =======================
st.markdown("""
    <style>
        h1 { font-size: 4rem !important; text-align: center; color: #4B0082; font-weight: bold; }
        h2, h3, h4 { font-size: 2.5rem !important; font-weight: bold; }
        label, .stSlider label, .stSelectbox label { font-size: 2rem !important; font-weight: bold; }
        .stTextInput input, .stNumberInput input, .stSelectbox div, .stSlider { font-size: 2rem !important; height: 3.5rem !important; }
        .stTextInput, .stNumberInput, .stSelectbox, .stSlider { width: 100% !important; }
        .stAlert { font-size: 2rem !important; }
        .big-result { font-size: 2.5rem !important; font-weight: bold; color: #333333; }
        table { font-size: 1.8rem !important; }
    </style>
""", unsafe_allow_html=True)

# =======================
# Header
# =======================
st.markdown("<h1>Credit Card Approval Predictor 💳</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:2rem;'>Enter applicant details to predict credit card approval</p>", unsafe_allow_html=True)
st.write("---")

# =======================
# Input Form
# =======================
with st.form("applicant_form"):
    st.subheader("Applicant Information")

    # Numeric Inputs
    age = st.slider("Age", 0, 100, 25)
    income = st.number_input("Income ($)", 0.0, 1000000.0, 3000.0, step=100.0, format="%.2f")
    credit_score = st.slider("Credit Score", 0, 20, 6)
    debt = st.slider("Existing Debt", 0.0, 30.0, 0.0, step=0.1)
    years_employed = st.number_input("Years Employed", 0.0, 50.0, 1.0, step=0.1, format="%.1f")

    # Categorical Inputs
    user_cats = {}
    for feature, options in categorical_features.items():
        user_cats[feature] = st.selectbox(feature, options)

    submitted = st.form_submit_button("Predict")

# =======================
# Prepare Input & Predict
# =======================
if submitted:
    user_input = {
        "Age": age,
        "Income": income,
        "CreditScore": credit_score,
        "Debt": debt,
        "YearsEmployed": years_employed
    }
    user_input.update(user_cats)

    input_df = pd.DataFrame([user_input])
    input_df = pd.get_dummies(input_df)

    # Ensure all required columns exist
    for col in X_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[X_columns]

    # Age validation
    if age < 18:
        st.error("❌ Prediction: Rejected")
        st.info("Reason: Applicant must be at least 18 years old.")
    elif age > 65:
        st.error("❌ Prediction: Rejected")
        st.info("Reason: Age limit for credit card exceeded.")
    else:
        # Model Predictions
        gam_probs = gam_model.predict_proba(input_df.values)
        # Handle 1D or 2D output
        if gam_probs.ndim == 1:
            gam_prob = gam_probs[0]
        else:
            gam_prob = gam_probs[0][1]

        

    

        prediction = "Approved ✅" if gam_prob > 0.5 else "Rejected ❌"

        # Display Results
        if prediction.startswith("Approved"):
            st.success(f"Prediction: {prediction}")
        else:
            st.error(f"Prediction: {prediction}")

        st.markdown(f"<p class='big-result'>Approval Probability (Ensemble): {gam_prob:.2f}</p>", unsafe_allow_html=True)
        st.write("---")

        # Applicant Summary
        st.subheader("Applicant Details")
        input_summary = pd.DataFrame(user_input.items(), columns=["Feature", "Value"])
        st.table(input_summary)
