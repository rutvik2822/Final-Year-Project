from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
from preprocess import preprocess_data  # Import the preprocessing function
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # ✅ Now correctly applied after app initialization

# Load trained model with error handling
try:
    model = joblib.load("captcha_model.pkl")
    print("Model loaded successfully!")
except Exception as e:
    print(f" Error loading model: {e}")
    model = None  # Avoid crashing the server if model load fails

@app.route('/predict', methods=['POST'])
def predict():
    print(" Request received!")  # Debugging print

    # Get JSON data
    data = request.get_json()
    print("Received Data:", data)  # Debugging print

    if not data or "drawing_data" not in data:
        print(" Error: No 'drawing_data' in request!")  # Debugging
        return jsonify({"error": "Invalid input format"}), 400

    # Extract features
    try:
        features = preprocess_data(data["drawing_data"])
        feature_names = ["average_speed", "speed_variance", "num_pauses", "total_time"]
        input_df = pd.DataFrame([features], columns=feature_names)

        # Predict using trained model
        prediction = model.predict(input_df)[0]
        print(f" Prediction: {'Human' if prediction == 0 else 'Bot'}")  # Debugging

        return jsonify({"prediction": "Human" if prediction == 0 else "Bot"})
    
    except Exception as e:
        print(f" Error during prediction: {e}")
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # ✅ Required for Render
    app.run(host='0.0.0.0', port=port, debug=True)





