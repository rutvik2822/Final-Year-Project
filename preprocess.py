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
    
    # Define thresholds
    min_distance_threshold = 2  # Ignore distances smaller than 2 px
    max_speed_threshold = 10  # Max reasonable speed (adjustable)

    for i in range(1, len(drawing_data)):
        x1, y1, t1 = drawing_data[i - 1]
        x2, y2, t2 = drawing_data[i]

        # Ensure valid time difference
        time_diff = t2 - t1
        if time_diff <= 0:
            continue  # Ignore incorrect timestamps

        # Calculate speed
        distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        if distance < min_distance_threshold:
            continue  # Ignore micro-movements

        speed = distance / time_diff
        speed = min(speed, max_speed_threshold)  # Cap max speed
        speeds.append(speed)

        print(f"🔹 Point {i}: ({x1},{y1}) -> ({x2},{y2}) | Time: {t1} -> {t2} | Distance: {distance:.3f} | Speed: {speed:.3f}")

        # Calculate pauses (if time difference is too large)
        avg_time_diff = (timestamps[-1] - timestamps[0]) / len(drawing_data)
        if time_diff > avg_time_diff * 2:
            pauses.append(time_diff)
    
    # Calculate features
    avg_speed = np.mean(speeds) if speeds else 0
    speed_variance = np.var(speeds) if speeds else 0
    
    features = {
        "average_speed": avg_speed,
        "speed_variance": speed_variance,
        "num_pauses": len(pauses),
        "total_time": timestamps[-1] - timestamps[0] if timestamps else 0
    }
    
    print(f"🔹 Final Features: {features}")
    return features

