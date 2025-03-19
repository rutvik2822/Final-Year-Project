import pandas as pd
import joblib

# Load dataset and model
df = pd.read_csv("drawing_data.csv")
model = joblib.load("captcha_model.pkl")

# Extract only bot samples (label = 1)
bot_samples = df[df["label"] == 1].drop(columns=["label"])

if bot_samples.empty:
    print("⚠️ No bot data found in the dataset!")
else:
    # Get probability of being classified as bot
    probabilities = model.predict_proba(bot_samples)[:, 1]  # Probability of bot (1)
    predictions = model.predict(bot_samples)  # Get actual predictions

    print("\n🔹 **First 10 Bot Prediction Probabilities:**")
    print(probabilities[:10])  # Show first 10 probabilities

    print("\n✅ **Predictions for Bot Data:**")
    print(predictions[:10])  # Show first 10 predictions
