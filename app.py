from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
from preprocess import preprocess_data  
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load trained model
model = joblib.load("captcha_model.pkl")

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

@app.route('/predict', methods=['POST'])
def predict():
    print("🔹 Request received!")

    data = request.get_json()
    print("🔹 Received Data:", data)

    if not data or "drawing_data" not in data:
        print("❌ Error: Invalid input format")
        return jsonify({"error": "Invalid input format"}), 400

    try:
        # Preprocess user input
        features = preprocess_data(data["drawing_data"])
        print(f"🔹 Extracted Features: {features}") 
        # Ensure correct feature names
        feature_names = ["average_speed", "speed_variance", "num_pauses", "total_time"]
        input_df = pd.DataFrame([features], columns=feature_names)

        # Get prediction probabilities
        probabilities = model.predict_proba(input_df)[0]  # Returns array of probabilities

        # Get predicted class
        predicted_class = model.classes_[np.argmax(probabilities)]

        # Log the confidence scores
        print(f"🔹 Prediction Confidence: {probabilities.tolist()} | Predicted: {predicted_class}")

        return jsonify({
            "prediction": "Human" if predicted_class == 0 else "Bot",
            "confidence": probabilities.tolist()  # Send confidence scores to frontend
        })
    
    except Exception as e:
        print(f"❌ Error in processing: {e}")
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == '__main__':
    app.run(debug=True)




