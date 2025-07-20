import pandas as pd

def preprocess_data(df):
    df = df.copy()

    # --- Step 1: Drop unnecessary columns ---
    df.drop(columns=['customerID'], inplace=True, errors='ignore')

    # --- Step 2: Standardize service column values ---
    service_cols = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines'
    ]
    for col in service_cols:
        if col in df.columns:
            df[col] = df[col].replace({
                'No internet service': 'No', 
                'No phone service': 'No'
            })

    # --- Step 3: Clean TotalCharges ---
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # --- Step 4: Feature Engineering ---
    if {'TotalCharges', 'tenure'}.issubset(df.columns):
        df['AvgChargesPerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)

    # --- Step 5: Encode binary columns ---
    binary_map = {'Yes': 1, 'No': 0}
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(binary_map)

    # --- Step 6: One-hot encode remaining categorical columns ---
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # --- Step 7: Final check ---
    df = df.fillna(0)
    return df