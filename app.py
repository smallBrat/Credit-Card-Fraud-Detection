import streamlit as st
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

# Import necessary libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Friendly labels for V1–V28
v_labels = {
    "V1": "Transaction Pattern Score 1",
    "V2": "Transaction Pattern Score 2",
    "V3": "Transaction Source Risk Factor",
    "V4": "Location Consistency Index",
    "V5": "Transaction Timing Behavior",
    "V6": "Merchant Trust Index",
    "V7": "Transaction Amount Influence",
    "V8": "Cardholder Behavior Score",
    "V9": "Transaction Channel Score",
    "V10": "Historical Match Factor",
    "V11": "Usage Pattern Factor",
    "V12": "Regional Usage Risk Index",
    "V13": "Merchant Frequency Score",
    "V14": "Cross-Border Risk Indicator",
    "V15": "Transaction Cluster Score",
    "V16": "User Spending Pattern Score",
    "V17": "Account Activity Score",
    "V18": "Spending Consistency Index",
    "V19": "Transaction Density Score",
    "V20": "Transaction Volume Score",
    "V21": "Merchant Match Score",
    "V22": "Device Trustworthiness Score",
    "V23": "Time-Based Risk Factor",
    "V24": "Account Stability Score",
    "V25": "Cross-Merchant Usage Indicator",
    "V26": "High-Risk Feature Index",
    "V27": "Transaction Deviation Score",
    "V28": "Unusual Pattern Score",
    "Amount": "Transaction Amount"
}

# ⬇️ Load model and scaler from files (updated here only)
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

# Get feature columns from training phase (must match original)
feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Load trained model and scaler
model, scaler = load_model()

# Streamlit UI
st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details below to check if it's **fraudulent** or **legitimate**.")

# Input fields with friendly names
with st.form("fraud_form"):
    user_input = {}
    for col in feature_cols:
        label = v_labels.get(col, col)  # Use friendly label if available
        user_input[col] = st.number_input(label, value=0.0, format="%.4f")
    submitted = st.form_submit_button("Check Fraud")

# Prediction
if submitted:
    input_df = pd.DataFrame([user_input])
    input_df['Amount'] = scaler.transform(input_df[['Amount']])
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraudulent transaction detected with **{proba*100:.2f}%** probability!")
    else:
        st.success(f"✅ Legitimate transaction with only **{proba*100:.2f}%** probability of fraud.")
