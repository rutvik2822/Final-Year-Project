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
    timestamps = [point[2] / 1000 for point in drawing_data]  # ✅ Convert ms → seconds
    
    # Define thresholds
    min_distance_threshold = 2  # Ignore very small movements
    max_speed_threshold = 1000  # Prevent unrealistic values

    for i in range(1, len(drawing_data)):
        x1, y1, t1 = drawing_data[i - 1]
        x2, y2, t2 = drawing_data[i]

        time_diff = (t2 / 1000) - (t1 / 1000)  # ✅ Convert timestamps to seconds
        if time_diff <= 0:
            continue

        # Calculate speed
        distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        if distance < min_distance_threshold:
            continue  

        speed = distance / time_diff
        speed = min(speed, max_speed_threshold)  
        speeds.append(speed)

        # Detect pauses (large time gaps)
        avg_time_diff = (timestamps[-1] - timestamps[0]) / len(drawing_data)
        if time_diff > avg_time_diff * 2:
            pauses.append(time_diff)
    
    avg_speed = np.mean(speeds) if speeds else 0
    speed_variance = np.var(speeds) if speeds else 0
    num_pauses = len(pauses)
    total_time = timestamps[-1] - timestamps[0] if timestamps else 0

    features = {
        "average_speed": avg_speed,
        "speed_variance": np.log1p(speed_variance),  # Log transform
        "num_pauses": num_pauses,
        "total_time": total_time
    }
    
    print(f"🔹 Fixed Features: {features}")  # ✅ Debugging
    return features

