import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset and model
df = pd.read_csv("drawing_data.csv")  
model = joblib.load("captcha_model.pkl")

# Separate features and labels
X = df.drop(columns=["label"])  
y_true = df["label"]

# Make predictions
y_pred = model.predict(X)

# Evaluate Model Performance
print("🔹 **Testing Model on Training Data**")
print(f"✅ Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print("\n🎯 Classification Report:\n", classification_report(y_true, y_pred))
