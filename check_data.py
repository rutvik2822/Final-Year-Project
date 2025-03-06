import pandas as pd

# Load the CSV file
data = pd.read_csv('drawing_data.csv')

# Check the label distribution (human vs. bot)
print(data['label'].value_counts())

# Check the first few rows of the dataset
print(data.head())
