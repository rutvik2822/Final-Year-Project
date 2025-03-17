import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
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
model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning using Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_

# Train the best model with the training data
best_model.fit(X_train, y_train)

# Check the model's accuracy on the test data
y_pred = best_model.predict(X_test)

# Evaluate the model
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model to a file
joblib.dump(best_model, 'captcha_model.pkl')
print("Model saved as 'captcha_model.pkl'")