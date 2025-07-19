import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE

# ✅ 1. Load the Data
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
# ✅ 2. Drop Unnecessary Columns
# 'customerID' is just an identifier, not useful for ML
df.drop('customerID', axis=1, inplace=True)

# ✅ 3. Convert 'TotalCharges' to numeric and handle missing values using KNN imputation
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Use KNNImputer for more accurate imputation of missing 'TotalCharges'
imputer = KNNImputer(n_neighbors=5)
df[['tenure', 'MonthlyCharges', 'TotalCharges']] = imputer.fit_transform(df[['tenure', 'MonthlyCharges', 'TotalCharges']])

# ✅ 4. Encode Categorical Variables

# Label encode binary categorical columns (Yes/No or Male/Female)
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn', 'gender']
for col in binary_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# One-hot encode remaining categorical columns (e.g., Contract, InternetService)
df = pd.get_dummies(df, drop_first=True)  # drop_first avoids dummy variable trap

# ✅ 5. Feature Scaling
# Standardize continuous variables to have zero mean and unit variance
scaler = StandardScaler()
df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df[['tenure', 'MonthlyCharges', 'TotalCharges']])

# ✅ 6. Separate Features and Target Variable
X = df.drop('Churn', axis=1)  # Independent features
y = df['Churn']               # Target: 0 = Not Churned, 1 = Churned

# ✅ 7. Handle Class Imbalance with SMOTE (Synthetic Minority Over-sampling Technique)
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)  # X_res, y_res are the balanced data