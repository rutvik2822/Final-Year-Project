from flask import Flask, request, jsonify
import pandas as pd
import joblib
from preprocess import preprocess_data  # Import the preprocessing function

app = Flask(__name__)

# Load trained model
model = joblib.load("captcha_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    print("Request received!")  # Debugging print

    # Get JSON data
    data = request.get_json()
    print(" Received Data:", data)  # Debugging print

    if not data or "drawing_data" not in data:
        print(" Error: No 'drawing_data' in request!")  # Debugging
        return jsonify({"error": "Invalid input format"}), 400

    # Extract features
    features = preprocess_data(data["drawing_data"])
    feature_names = ["average_speed", "speed_variance", "num_pauses", "total_time"]
    input_df = pd.DataFrame([features], columns=feature_names)

    # Predict using trained model
    prediction = model.predict(input_df)[0]
    print(f" Prediction: {'Human' if prediction == 0 else 'Bot'}")  # Debugging

    return jsonify({"prediction": "Human" if prediction == 0 else "Bot"})

@app.route('/')
def home():
    return "Flask App is Running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)






