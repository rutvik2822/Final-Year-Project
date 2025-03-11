from flask import Flask, request, jsonify
import pandas as pd
import joblib
from preprocess import preprocess_data  
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load trained model
model = joblib.load("captcha_model.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "CAPTCHA Prediction API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    print("Request received!")  

    data = request.get_json()
    print(" Received Data:", data)  

    if not data or "drawing_data" not in data:
        return jsonify({"error": "Invalid input format"}), 400

    features = preprocess_data(data["drawing_data"])
    feature_names = ["average_speed", "speed_variance", "num_pauses", "total_time"]
    input_df = pd.DataFrame([features], columns=feature_names)

    prediction = model.predict(input_df)[0]
    return jsonify({"prediction": "Human" if prediction == 0 else "Bot"})

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))  # Ensure PORT is set to 10000
    app.run(debug=True, host="0.0.0.0", port=port)




