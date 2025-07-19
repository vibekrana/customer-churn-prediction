import pandas as pd
import joblib
from src.preprocess import preprocess_data

# Load the trained model
model = joblib.load('models/model.pkl')

# Example input data (replace with actual user input)
input_data = {
    'tenure': 12,
    'MonthlyCharges': 50.0,
    'TotalCharges': 600.0,
    'Partner': 'No',
    'Dependents': 'No',
    'PhoneService': 'Yes',
    'PaperlessBilling': 'Yes',
    'Churn': 'No',
    'gender': 'Male'
}
input_df = pd.DataFrame([input_data])

# Preprocess the input data
processed_input = preprocess_data(input_df)

# Make prediction
prediction = model.predict(processed_input)
churn_status = 'Will Churn' if prediction[0] == 1 else 'Will Not Churn'
print(f'Prediction: The customer {churn_status}.')