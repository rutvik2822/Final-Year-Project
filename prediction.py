import pandas as pd
import joblib

# Load the dataset
df = pd.read_csv("drawing_data.csv")  # Ensure this file exists

# Define input features (X) by dropping the label column
X = df.drop(columns=["label"])

# Load the trained model
model = joblib.load("captcha_model.pkl")

# Check feature importance
if hasattr(model, "feature_importances_"):
    feature_importance = model.feature_importances_
    feature_names = X.columns

    sorted_idx = feature_importance.argsort()[::-1]
    for idx in sorted_idx:
        print(f"{feature_names[idx]}: {feature_importance[idx]:.4f}")
else:
    print("Feature importance not available for this model.")

# **🔹 Use Probabilities for Bot Detection**
probabilities = model.predict_proba(X)[:, 1]  # Probability of being a bot

# **🔹 Debug: Print Probabilities**
print("\n🔹 **First 10 Prediction Probabilities:**")
print(probabilities[:10])

# **🔹 Adjust Decision Threshold**
threshold = 0.2  # Try lowering further if bots are not detected

predictions = (probabilities > threshold).astype(int)

# **🔹 Display Results**
print("\n✅ Predictions:", predictions[:10])  # Show first 10 predictions
print("🎯 True Labels:", df["label"][:10].values)  # Compare with actual labels

