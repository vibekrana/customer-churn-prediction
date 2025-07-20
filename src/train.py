import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import pickle
import os
from preprocess import preprocess_data

def train_model():
    # --- Load and preprocess data ---
    # Use correct path for CSV file
    csv_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if not os.path.exists(csv_path):
        csv_path = "../WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    df = pd.read_csv(csv_path)
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

    # --- Create models directory if it doesn't exist ---
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # --- Save model with correct path ---
    model_path = os.path.join(models_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to {model_path}")

    # --- Plot feature importance ---
    feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)[:15]
    plt.figure(figsize=(8, 6))
    feat_imp.plot(kind="barh")
    plt.title("Top 15 Features - Random Forest")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Save plot with correct path
    plot_path = os.path.join(models_dir, "feature_importance.png")
    plt.savefig(plot_path)
    print(f"📈 Feature importance saved to {plot_path}")
    plt.close()

if __name__ == "__main__":
    train_model()