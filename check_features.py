import pandas as pd

# Load the dataset
df = pd.read_csv("drawing_data.csv")

# Separate human and bot data
human_data = df[df["label"] == 0]
bot_data = df[df["label"] == 1]

# Compute statistics for both
feature_summary = pd.DataFrame({
    "Feature": ["average_speed", "speed_variance", "num_pauses", "total_time"],
    "Human Mean": [
        human_data["average_speed"].mean(),
        human_data["speed_variance"].mean(),
        human_data["num_pauses"].mean(),
        human_data["total_time"].mean()
    ],
    "Bot Mean": [
        bot_data["average_speed"].mean(),
        bot_data["speed_variance"].mean(),
        bot_data["num_pauses"].mean(),
        bot_data["total_time"].mean()
    ],
    "Human Std": [
        human_data["average_speed"].std(),
        human_data["speed_variance"].std(),
        human_data["num_pauses"].std(),
        human_data["total_time"].std()
    ],
    "Bot Std": [
        bot_data["average_speed"].std(),
        bot_data["speed_variance"].std(),
        bot_data["num_pauses"].std(),
        bot_data["total_time"].std()
    ],
})

# Print feature comparison
print("🔹 Feature Comparison Between Human and Bot Data:")
print(feature_summary)
