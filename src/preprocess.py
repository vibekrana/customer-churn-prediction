import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE

def preprocess_data(df):
    # ✅ 2. Drop Unnecessary Columns (handle missing 'customerID' gracefully)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1, inplace=False)
    else:
        df = df.copy()  # Create a copy if no 'customerID' to avoid modifying original

    # ✅ 3. Convert 'TotalCharges' to numeric and handle missing values using KNN imputation
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Use KNNImputer for more accurate imputation of missing 'TotalCharges'
    imputer = KNNImputer(n_neighbors=5)
    df[['tenure', 'MonthlyCharges', 'TotalCharges']] = imputer.fit_transform(df[['tenure', 'MonthlyCharges', 'TotalCharges']])

    # ✅ 4. Encode Categorical Variables
    # Label encode binary categorical columns (Yes/No or Male/Female)
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn', 'gender']
    for col in binary_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col])

    # One-hot encode remaining categorical columns (e.g., Contract, InternetService)
    df = pd.get_dummies(df, drop_first=True)  # drop_first avoids dummy variable trap

    # ✅ 5. Feature Scaling
    scaler = StandardScaler()
    df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df[['tenure', 'MonthlyCharges', 'TotalCharges']])

    # ✅ 6. Separate Features and Target Variable
    X = df.drop('Churn', axis=1, errors='ignore')  # Ignore if 'Churn' not present
    y = df['Churn'] if 'Churn' in df.columns else pd.Series([0])  # Default to 0 if not present

    # ✅ 7. Handle Class Imbalance with SMOTE (skip if y is empty or single value)
    if len(y) > 1 and y.nunique() > 1:
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
    else:
        X_res, y_res = X, y

    return X_res, y_res