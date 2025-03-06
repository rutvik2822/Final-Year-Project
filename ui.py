import tkinter as tk
import time
import csv
import os
import joblib
import pandas as pd
from preprocess import preprocess_data  # Import the preprocessing function

# Load the trained model
model = joblib.load('captcha_model.pkl')

# Path to the CSV file where we'll save human data
CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), "drawing_data.csv")

# Function to start drawing (initialize coordinates and timestamp)
def start_drawing(event):
    canvas.old_coords = (event.x, event.y, time.time())
    canvas.draw_data = [(event.x, event.y, time.time())]  # Initialize drawing data

# Function to draw on the canvas and capture speed
def draw(event):
    if not hasattr(canvas, 'old_coords'):
        return
    x, y = event.x, event.y
    x1, y1, t1 = canvas.old_coords
    t2 = time.time()
    canvas.create_line(x1, y1, x, y, width=2, fill='black', capstyle=tk.ROUND, smooth=True)
    canvas.old_coords = (x, y, t2)
    canvas.draw_data.append((x, y, t2))  # Append position and timestamp

# Function to calculate pauses (lower threshold for touchpad sensitivity)
def calculate_pauses(data, threshold=0.02):  # Reduced from 0.05 to 0.02
    pauses = 0
    for i in range(1, len(data)):
        if data[i][2] - data[i-1][2] > threshold:
            pauses += 1
    return pauses

# Function to reset drawing when mouse is released
def reset_drawing(event):
    print(" reset_drawing() function called!")  # Debugging print

    if not hasattr(canvas, 'draw_data') or len(canvas.draw_data) < 2:
        print(" Not enough data to process!")
        return

    print("Resetting drawing and processing data...")  

    features = preprocess_data(canvas.draw_data)

    # Manually add `num_pauses` to features
    features['num_pauses'] = calculate_pauses(canvas.draw_data)

    prediction = make_prediction(features)

    print(f"Prediction: {prediction[0]}")  

    show_prediction_on_canvas(prediction[0])

    features['label'] = prediction[0]

    print(" Sending data to save_features_to_csv()")
    save_features_to_csv(features)  # This function should be called here!
    print(" Features successfully saved!")

# Function to save features to a CSV file
def save_features_to_csv(features):
    try:
        print(" save_features_to_csv() function called!")  # Debugging print
        print(" Attempting to save to:", CSV_FILE_PATH)  

        file_exists = os.path.exists(CSV_FILE_PATH)

        with open(CSV_FILE_PATH, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=features.keys())

            if not file_exists:
                print("Writing CSV headers...")
                writer.writeheader()

            writer.writerow(features)
            print(f"Data successfully saved: {features}")

    except Exception as e:
        print(f"Error saving data to CSV: {e}")

# Function to make predictions using the trained model
def make_prediction(features):
    feature_names = ['average_speed', 'speed_variance', 'num_pauses', 'total_time']
    feature_values = [list(features.values())]  # Convert the dictionary to a list
    feature_df = pd.DataFrame(feature_values, columns=feature_names)
    prediction = model.predict(feature_df)
    return prediction

# Function to show prediction on the canvas
def show_prediction_on_canvas(prediction):
    canvas.delete("prediction_text")
    x, y = 60, 280  # Bottom-left corner
    text = "Human" if prediction == 0 else "Bot"
    color = "green" if prediction == 0 else "red"
    canvas.create_text(x, y, text=text, fill=color, font=('Helvetica', 24), tags="prediction_text")
    root.update()

# Function to clear the canvas (only when button is clicked)
def clear_canvas():
    canvas.delete("all")  # Clear all drawings on the canvas
    canvas.draw_data = []  # Reset drawing data
    print("Canvas cleared!")

# Set up the main Tkinter window
root = tk.Tk()
root.title("Behavioral CAPTCHA - Collect Human Data")

# Create a canvas widget for drawing
canvas = tk.Canvas(root, width=400, height=300, bg="white")
canvas.pack()

# Bind mouse events for interaction
canvas.bind("<ButtonPress-1>", start_drawing)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", reset_drawing)

# Add a "Clear Canvas" button
clear_button = tk.Button(root, text="Clear Canvas", command=clear_canvas)
clear_button.pack()

# Run the Tkinter event loop
root.mainloop()

