import csv
import os
import random
import time

# Path to the CSV file where bot data will be saved
CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), "drawing_data.csv")

# Function to generate simulated bot data
def generate_bot_data(samples=200):
    bot_data = []
    for _ in range(samples):
        average_speed = random.uniform(500, 1000)  # Bots move faster
        speed_variance = random.uniform(10000, 30000)  # Less variation in speed
        num_pauses = random.randint(0, 1)  # Bots rarely pause
        total_time = random.uniform(0.5, 2)  # Faster completion time
        label = 1  # Label for bots

        bot_data.append({
            "average_speed": average_speed,
            "speed_variance": speed_variance,
            "num_pauses": num_pauses,
            "total_time": total_time,
            "label": label
        })

    return bot_data

# Function to save bot data to CSV
def save_bot_data(bot_data):
    file_exists = os.path.exists(CSV_FILE_PATH)
    
    with open(CSV_FILE_PATH, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=bot_data[0].keys())

        if not file_exists:
            print("Writing CSV headers...")
            writer.writeheader()

        writer.writerows(bot_data)
        print(f" Successfully added {len(bot_data)} bot samples to {CSV_FILE_PATH}")

# Generate and save bot data
bot_data = generate_bot_data(samples=200)
save_bot_data(bot_data)
