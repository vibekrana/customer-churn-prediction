import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import pickle
from src.preprocess import preprocess_data

def train_model():
    # --- Load and preprocess data ---
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = preprocess_data(df)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # --- Split and balance dataset ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

    # --- Train Random Forest ---
    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_train_bal, y_train_bal)

    # --- Evaluate ---
    y_pred = model.predict(X_test)
    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # --- Save model ---
    with open("../models/model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("✅ Model saved to ../models/model.pkl")

    # --- Plot feature importance ---
    feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)[:15]
    plt.figure(figsize=(8, 6))
    feat_imp.plot(kind="barh")
    plt.title("Top 15 Features - Random Forest")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("../models/feature_importance.png")
    print("📈 Feature importance saved to ../models/feature_importance.png")

if __name__ == "__main__":
    train_model()