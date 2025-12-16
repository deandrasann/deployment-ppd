from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load model and feature names
try:
    with open('stress_level_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    print("Model & Features loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    feature_names = []

# SESUAIKAN DENGAN DATASET (0, 1, 2)
STRESS_LEVEL_DESC = {
    0: "Low Stress - You are in a relaxed state. Keep up your healthy lifestyle!",
    1: "Medium Stress - You are experiencing moderate stress. Consider some relaxation or a short break.",
    2: "High Stress - Your stress level is high. Please consider consulting a professional or taking a long rest."
}

@app.route('/')
def home():
    return jsonify({
        "message": "Stress Level Prediction API (Logistic Regression Tuned)",
        "status": "active",
        "status": "active",
        "endpoints": {
            "/predict": "POST - Make stress level predictions",
            "/health": "GET - Check API health"
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server"}), 500
        
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Convert ke DataFrame
        input_df = pd.DataFrame([data])
        
        # Pastikan hanya fitur yang ada di feature_names yang masuk ke model
        # dan urutannya HARUS sama
        missing_features = [f for f in feature_names if f not in input_df.columns]
        if missing_features:
            # Jika ada fitur yang kurang dari kuesioner, isi dengan median/0
            for f in missing_features:
                input_df[f] = 0 
        
        # Susun ulang kolom sesuai saat training
        input_df = input_df[feature_names].astype(float)
        
        # Prediksi
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Output respons
        return jsonify({
            "prediction": int(prediction),
            "description": STRESS_LEVEL_DESC.get(int(prediction), "Unknown stress level"),
            "confidence": round(float(np.max(prediction_proba)) * 100, 2), # Keyakinan tertinggi dalam %
            "probabilities": {
                "low": float(prediction_proba[0]),
                "medium": float(prediction_proba[1]),
                "high": float(prediction_proba[2])
            },
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)