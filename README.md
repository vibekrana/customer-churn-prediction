🔮 Customer Churn Prediction App
A machine learning web application that predicts customer churn using Random Forest algorithm.

🚀 Live Demo
Deploy this app on Render

📋 Features
Machine Learning Model: Random Forest with SMOTE balancing
Interactive Web Interface: Beautiful, responsive UI
Real-time Predictions: Instant churn risk assessment
Confidence Scoring: Prediction confidence levels
Feature Engineering: Advanced data preprocessing
🛠️ Tech Stack
Backend: Flask, Python
ML Libraries: scikit-learn, pandas, numpy
Data Balancing: imbalanced-learn (SMOTE)
Frontend: HTML5, CSS3, JavaScript
Deployment: Render, Gunicorn
📁 Project Structure
customer-churn-prediction/
├── Procfile                 # Render deployment config
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
└── src/
    ├── __init__.py         # Python package init
    ├── app.py              # Flask web application
    ├── train.py            # Model training script
    ├── preprocess.py       # Data preprocessing
    ├── models/
    │   ├── model.pkl       # Trained model
    │   └── feature_importance.png
    └── templates/
        └── index.html      # Web interface
🏃‍♂️ Quick Start
1. Clone & Setup
bash
git clone <your-repo>
cd customer-churn-prediction
pip install -r requirements.txt
2. Train Model
bash
cd src
python train.py
3. Run Locally
bash
python app.py
Visit http://localhost:5000

🌐 Deploy to Render
Push to GitHub
Create Render Web Service
Build Command: pip install -r requirements.txt
Start Command: cd src && gunicorn app:app
📊 Model Performance
Algorithm: Random Forest (150 trees, max_depth=10)
Data Balancing: SMOTE oversampling
Features: 20+ engineered features
Accuracy: ~79% on test set
🎯 API Endpoints
GET / - Web interface
POST /predict - Prediction API
GET /health - Health check
📱 Sample API Request
json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 50.0
}
📈 Model Features
Key factors in churn prediction:

Contract type (month-to-month higher risk)
Tenure (longer tenure = lower churn)
Monthly charges
Payment method
Internet service type
Demographics (senior citizen, partner, dependents)
🔧 Development
Data Preprocessing
Handle missing values
Feature engineering (AvgChargesPerMonth)
Binary encoding for Yes/No fields
One-hot encoding for categorical variables
SMOTE for class imbalance
Model Training
Random Forest Classifier
Grid search for hyperparameters
Cross-validation
Feature importance analysis
📄 License
This project is open source and available under the MIT License.

🤝 Contributing
Fork the project
Create your feature branch
Commit your changes
Push to the branch
Open a Pull Request
⭐ Star this repo if you found it helpful!

