from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from preprocess import preprocess_data
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = pickle.load(open("../models/model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_df = pd.DataFrame([data])
    processed_df = preprocess_data(input_df)

    model_features = model.feature_names_in_
    for col in model_features:
        if col not in processed_df.columns:
            processed_df[col] = 0
    processed_df = processed_df[model_features]

    prediction = model.predict(processed_df)[0]
    confidence = model.predict_proba(processed_df)[0][1]

    return jsonify({
        "prediction": int(prediction),
        "confidence": round(confidence, 2)
    })

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)