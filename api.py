from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and feature names
try:
    with open('stress_level_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    print("Model loaded successfully!")
except:
    print("Warning: Model not found. Please run train_model.py first.")
    model = None
    feature_names = []

# Stress level descriptions
STRESS_LEVEL_DESC = {
    0: "No Stress - You're doing great! Your stress levels are minimal.",
    1: "Mild Stress - You're experiencing some stress but managing well.",
    2: "Moderate Stress - You're under noticeable stress. Consider taking breaks and practicing relaxation techniques.",
    3: "High Stress - You're experiencing significant stress. It's important to seek support and practice self-care.",
    4: "Severe Stress - You're under extreme stress. Please consider seeking professional help immediately."
}

@app.route('/')
def home():
    return jsonify({
        "message": "Stress Level Prediction API",
        "status": "active",
        "endpoints": {
            "/predict": "POST - Make stress level predictions",
            "/health": "GET - Check API health"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Convert to DataFrame with correct feature names
        input_df = pd.DataFrame([data])
        
        # Ensure all features are present
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0  # Default value
        
        # Reorder columns to match training
        input_df = input_df[feature_names]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Get probability for each class
        probabilities = {
            f"stress_level_{i}": float(prediction_proba[i]) 
            for i in range(len(prediction_proba))
        }
        
        # Get description
        description = STRESS_LEVEL_DESC.get(int(prediction), "Unknown stress level")
        
        return jsonify({
            "prediction": int(prediction),
            "stress_level": int(prediction),
            "description": description,
            "probabilities": probabilities,
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/features', methods=['GET'])
def get_features():
    return jsonify({
        "features": feature_names,
        "count": len(feature_names)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)