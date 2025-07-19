import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    df = df.copy()

    # Drop customerID if present
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Replace 'No internet service' and 'No phone service' with 'No'
    replace_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies', 'MultipleLines']
    for col in replace_cols:
        if col in df.columns:
            df[col] = df[col].replace({'No internet service': 'No', 'No phone service': 'No'})

    # Convert TotalCharges to numeric
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Encode binary categorical features
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    le = LabelEncoder()
    for col in binary_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    # One-hot encode other categorical features
    df = pd.get_dummies(df, drop_first=True)

    return df
