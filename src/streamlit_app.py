import streamlit as st
import pandas as pd
import joblib
from preprocess import preprocess_data
import numpy as np

# Load the trained model
model = joblib.load('../models/model.pkl')

# Get the feature names the model was trained on
trained_feature_names = model.feature_names_in_

# Title of the app
st.title('Customer Churn Prediction (INR)')

# Header and instructions
st.header('Enter Customer Details')
st.write('All monetary values are in Indian Rupees (INR).')

# Input fields for customer data
tenure = st.number_input('Tenure (months)', min_value=0, max_value=100, value=0)
monthly_charges = st.number_input('Monthly Charges (INR)', min_value=0.0, max_value=10000.0, value=0.0, format="%.2f")
total_charges = st.number_input('Total Charges (INR)', min_value=0.0, max_value=100000.0, value=0.0, format="%.2f")
gender = st.selectbox('Gender', ['Female', 'Male'])
partner = st.selectbox('Has Partner', ['No', 'Yes'])
dependents = st.selectbox('Has Dependents', ['No', 'Yes'])
phone_service = st.selectbox('Phone Service', ['No', 'Yes'])
paperless_billing = st.selectbox('Paperless Billing', ['No', 'Yes'])
contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])

# Create a dictionary with input data
input_data = {
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'gender': gender,
    'Partner': partner,
    'Dependents': dependents,
    'PhoneService': phone_service,
    'PaperlessBilling': paperless_billing,
    'Contract': contract,
    'InternetService': internet_service
}
input_df = pd.DataFrame([input_data])

# Preprocess the input data
X_processed, _ = preprocess_data(input_df)

# Align columns with training data
for col in trained_feature_names:
    if col not in X_processed.columns:
        X_processed[col] = 0
st.write("Processed columns:", X_processed.columns.tolist())  # Debug output

# Reorder and subset to trained feature names
X_processed = X_processed[trained_feature_names]

# Make prediction with probability
prediction_proba = model.predict_proba(X_processed)
prediction = (prediction_proba[:, 1] >= 0.5).astype(int)  # Default threshold 0.5

# Display prediction
if st.button('Predict'):
    churn_status = 'Will Churn' if prediction[0] == 1 else 'Will Not Churn'
    st.write(f'Prediction: The customer {churn_status}.')
    st.write(f'Churn Probability: {prediction_proba[0, 1]:.2f}')  # Show probability