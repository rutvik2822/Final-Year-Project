from flask import Flask, request, jsonify
import pandas as pd
import joblib
from preprocess import preprocess_data  # Import the preprocessing function
from flask_cors import CORS

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Move after app initialization

# Load trained model
try:
    model = joblib.load("captcha_model.pkl")
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)

# ✅ Fix for the 404 Error (Home Route)
@app.route('/')
def home():
    return jsonify({"message": "CAPTCHA Prediction API is running!"})

# ✅ Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    print("Request received!")  # Debugging print

    # Get JSON data
    data = request.get_json()
    print("Received Data:", data)  # Debugging print

    if not data or "drawing_data" not in data:
        print("Error: No 'drawing_data' in request!")  # Debugging
        return jsonify({"error": "Invalid input format"}), 400

    # Extract features
    features = preprocess_data(data["drawing_data"])
    feature_names = ["average_speed", "speed_variance", "num_pauses", "total_time"]
    input_df = pd.DataFrame([features], columns=feature_names)

    # Predict using trained model
    prediction = model.predict(input_df)[0]
    print(f"Prediction: {'Human' if prediction == 0 else 'Bot'}")  # Debugging

    return jsonify({"prediction": "Human" if prediction == 0 else "Bot"})

# ✅ Ensure Correct Port Usage
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=10000)  # Ensure correct port usage




