import streamlit as st
import numpy as np
import pandas as pd
import base64
import pickle

# Load model
model = pickle.load(open('loan_status_model.pkl', 'rb'))

# Page config
st.set_page_config(page_title="Loan Prediction App", layout="wide")

# === Function to add a background image === #
def add_bg_from_local(image_file):
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add the background image
add_bg_from_local('loan_image.jpg')  # Use your image file name

# Centered title
st.markdown("<h1 style='text-align: center; color: black;'>Loan Application Evaluator</h1>", unsafe_allow_html=True)

# Two-column layout
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Marital Status", ["No", "Yes"])
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=4)
    education = st.selectbox("Education", ["Not Graduate", "Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

with col2:
    applicant_income = st.number_input("Applicant Income (in thousands)", min_value=0, value=0)
    coapplicant_income = st.number_input("Co-applicant Income (in thousands)", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=0)
    loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=0, value=0)
    credit_history = st.selectbox("Credit History (0 = No history, 1 = Good history)", [0, 1])

# Dictionary of input
user_data = {
    'Gender': gender,
    'Married': married,
    'Dependents': dependents,
    'Education': education,
    'Self_Employed': self_employed,
    'ApplicantIncome': applicant_income,
    'CoapplicantIncome': coapplicant_income,
    'LoanAmount': loan_amount,
    'Loan_Amount_Term': loan_amount_term,
    'Credit_History': credit_history,
    'Property_Area': property_area
}

# Preprocessing
def preprocess_features(features):
    features.interpolate(method='linear', inplace=True)
    features['Gender'].fillna(features['Gender'].mode()[0], inplace=True)
    features['Married'].fillna(features['Married'].mode()[0], inplace=True)
    features['Dependents'].fillna(features['Dependents'].mode()[0], inplace=True)
    features['Self_Employed'].fillna(features['Self_Employed'].mode()[0], inplace=True)

    features.replace({'Married': {'No': 0, 'Yes': 1},
                      'Gender': {'Male': 1, 'Female': 0},
                      'Self_Employed': {'No': 0, 'Yes': 1},
                      'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
                      'Education': {'Graduate': 1, 'Not Graduate': 0}}, inplace=True)

    features = features.replace(to_replace='3+', value=4)
    return features

processed_data = preprocess_features(pd.DataFrame(user_data, index=[0]))

# Prediction
if st.button("Predict Loan Status"):
    prediction = model.predict(processed_data)

    if prediction[0] == 1:
        st.success("🎉 Congratulations! Your loan is likely to be approved.")
    else:
        st.error(" Sorry, your loan is likely to be rejected.")
