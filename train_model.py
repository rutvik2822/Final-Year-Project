import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the dataset (CSV file with features and labels)
data = pd.read_csv('drawing_data.csv')

# Check if the dataset is loaded correctly
print(data.head())  # Show the first few rows of the dataset

# Features are all columns except 'label'
X = data.drop(columns=['label'])

# Labels are the 'label' column
y = data['label']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shape of the splits
print(f'Training data size: {X_train.shape[0]}')
print(f'Testing data size: {X_test.shape[0]}')

# Initialize the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model with the training data
model.fit(X_train, y_train)

# Check the model's accuracy on the test data
y_pred = model.predict(X_test)

# Evaluate the model
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model to a file
joblib.dump(model, 'captcha_model.pkl')
print("Model saved as 'captcha_model.pkl'")
