import pickle
import pandas as pd
from preprocess import preprocess_data

def predict(input_dict):
    model = pickle.load(open("../models/model.pkl", "rb"))
    input_df = pd.DataFrame([input_dict])
    processed_df = preprocess_data(input_df)

    # Ensure the input columns match training columns
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in processed_df.columns:
            processed_df[col] = 0
    processed_df = processed_df[model_features]

    prediction = model.predict(processed_df)
    return prediction[0]
