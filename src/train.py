import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from preprocess import preprocess_data

# Create models directory if it doesn't exist
os.makedirs('../models', exist_ok=True)

# Load and preprocess the data
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
X, y = preprocess_data(df)

# Print class distribution before and after SMOTE
print("Original class distribution:", y.value_counts())
print("Processed class distribution:", pd.Series(y).value_counts())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print("Classification Report:\n", classification_report(y_test, y_pred))

# Print prediction distribution
print("Predicted class distribution:", pd.Series(y_pred).value_counts())

# Save the model
joblib.dump(model, '../models/model.pkl')