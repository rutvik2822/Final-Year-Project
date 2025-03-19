import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load("captcha_model.pkl")

# Define **real-world test cases** (simulate user input)
test_samples = pd.DataFrame([
    {"average_speed": 300, "speed_variance": 40000, "num_pauses": 10, "total_time": 3},
    {"average_speed": 700, "speed_variance": 20000, "num_pauses": 0, "total_time": 1},
    {"average_speed": 150, "speed_variance": 50000, "num_pauses": 30, "total_time": 5},
    {"average_speed": 900, "speed_variance": 15000, "num_pauses": 2, "total_time": 2}
])

# Get model predictions
predictions = model.predict(test_samples)
prediction_probs = model.predict_proba(test_samples)

# Display results
for i, (pred, prob) in enumerate(zip(predictions, prediction_probs)):
    print(f"Sample {i+1}: Predicted {'Human' if pred == 0 else 'Bot'} (Confidence: {prob})")
