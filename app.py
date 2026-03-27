import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load files
model = pickle.load(open('loan_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
feature_columns = pickle.load(open('feature_columns.pkl', 'rb'))

st.title("💰 Loan Risk Prediction App")
st.write("Enter customer details")

# Inputs
credit_score = st.number_input("Credit Score", 300, 850)
income = st.number_input("Annual Income")
loan_amount = st.number_input("Loan Amount")
dti = st.number_input("Debt to Income Ratio")
interest_rate = st.number_input("Interest Rate")
delinq = st.number_input("Number of Delinquencies")

# Create dataframe
input_data = pd.DataFrame(columns=feature_columns)
input_data.loc[0] = 0

# Assign values
input_data['credit_score'] = credit_score
input_data['annual_income'] = income
input_data['loan_amount'] = loan_amount
input_data['debt_to_income_ratio'] = dti
input_data['interest_rate'] = interest_rate
input_data['num_of_delinquencies'] = delinq

# Feature engineering
input_data['income_loan_ratio'] = income / (loan_amount + 1)
input_data['risk_score'] = dti * interest_rate
input_data['credit_strength'] = credit_score / 850

# Scale
features_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    if prediction == 1:
        st.success(f"✅ Low Risk (Repayment Probability: {probability:.2f})")
    else:
        st.error(f"⚠️ High Risk (Default Probability: {1-probability:.2f})")