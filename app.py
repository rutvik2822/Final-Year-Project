from flask import Flask, request, jsonify
import pandas as pd
import joblib
from preprocess import preprocess_data  
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load trained model
model = joblib.load("captcha_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    print("Request received!")

    data = request.get_json()
    print("Received Data:", data)

    if not data or "drawing_data" not in data:
        return jsonify({"error": "Invalid input format"}), 400

    features = preprocess_data(data["drawing_data"])
    
    # Ensure correct feature names
    feature_names = ["average_speed", "speed_variance", "num_pauses", "total_time"]
    input_df = pd.DataFrame([features], columns=feature_names)

    prediction = model.predict(input_df)[0]
    print(f"Prediction: {'Human' if prediction == 0 else 'Bot'}")

    return jsonify({"prediction": "Human" if prediction == 0 else "Bot"})

if __name__ == '__main__':
    app.run(debug=True)




