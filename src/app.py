from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from preprocess import preprocess_data
from flask_cors import CORS
import os

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Load model with proper error handling
try:
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully")
except FileNotFoundError:
    print("❌ Model file not found. Please run train.py first.")
    model = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            "error": "Model not loaded. Please train the model first."
        }), 500
    
    try:
        data = request.json
        input_df = pd.DataFrame([data])
        processed_df = preprocess_data(input_df)

        # Ensure all model features are present
        model_features = model.feature_names_in_
        for col in model_features:
            if col not in processed_df.columns:
                processed_df[col] = 0
        
        # Reorder columns to match model training
        processed_df = processed_df[model_features]

        prediction = model.predict(processed_df)[0]
        confidence = model.predict_proba(processed_df)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "confidence": round(float(confidence), 2)
        })
    
    except Exception as e:
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)