import numpy as np

def preprocess_data(drawing_data):
    """
    Preprocess drawing data to extract meaningful features.
    :param drawing_data: List of (x, y, timestamp)
    :return: Dictionary of extracted features
    """
    if len(drawing_data) < 2:
        return {"average_speed": 0, "speed_variance": 0, "num_pauses": 0, "total_time": 0}

    speeds = []
    pauses = []
    timestamps = [point[2] for point in drawing_data]

    # Dynamic speed capping (based on percentiles)
    MAX_SPEED_THRESHOLD = 500  # Adjusted max speed based on observed data

    for i in range(1, len(drawing_data)):
        x1, y1, t1 = drawing_data[i - 1]
        x2, y2, t2 = drawing_data[i]

        # Ensure valid time difference
        time_diff = t2 - t1
        if time_diff <= 0:
            continue  # Ignore incorrect timestamps

        # Calculate speed
        distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        speed = distance / time_diff

        # Cap speed to remove extreme values
        speed = min(speed, MAX_SPEED_THRESHOLD)  
        speeds.append(speed)

        # Calculate pauses: If a time jump is **greater than 3 times** the avg time diff, it's a pause
        avg_time_diff = (timestamps[-1] - timestamps[0]) / len(drawing_data)
        if time_diff > avg_time_diff * 3:
            pauses.append(time_diff)

    # Compute final features
    avg_speed = np.mean(speeds) if speeds else 0
    speed_variance = np.var(speeds) if speeds else 0
    speed_variance = np.log1p(speed_variance)  # Log transformation

    features = {
        "average_speed": avg_speed,
        "speed_variance": speed_variance,
        "num_pauses": len(pauses),
        "total_time": timestamps[-1] - timestamps[0] if timestamps else 0
    }

    print(f"🔹 Extracted Features: {features}")
    return features


