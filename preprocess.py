import numpy as np

def preprocess_data(drawing_data):
    """
    Preprocess drawing data to extract meaningful features for ML analysis.
    :param drawing_data: List of tuples [(x, y, timestamp), ...]
    :return: Dictionary of extracted features
    """
    if len(drawing_data) < 2:
        # If only one point is drawn, return zero values
        return {"average_speed": 0, "speed_variance": 0, "num_pauses": 0, "total_time": 0}

    speeds = []
    pauses = []
    timestamps = [point[2] for point in drawing_data]
    
    for i in range(1, len(drawing_data)):
        x1, y1, t1 = drawing_data[i - 1]
        x2, y2, t2 = drawing_data[i]

        # Ensure time difference is not too small to avoid division by near-zero
        time_diff = max(t2 - t1, 0.01)  # Minimum threshold to avoid extreme values
        
        # Calculate speed
        distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        speed = distance / time_diff
        speeds.append(speed)

        # Calculate pauses dynamically
        avg_time_diff = (timestamps[-1] - timestamps[0]) / len(drawing_data)
        if time_diff > avg_time_diff * 2:  # Pause is detected if gap is twice the average interval
            pauses.append(time_diff)
    
    # Calculate average and variance of speed
    avg_speed = np.mean(speeds) if speeds else 0
    speed_variance = np.var(speeds) if speeds else 0
    
    # Feature dictionary
    features = {
        "average_speed": avg_speed,
        "speed_variance": speed_variance,
        "num_pauses": len(pauses),
        "total_time": timestamps[-1] - timestamps[0] if timestamps else 0
    }
    
    return features

# Example usage with dummy data
if __name__ == "__main__":
    # Sample drawing data (x, y, timestamp)
    sample_data = [
        (100, 200, 0.1),
        (102, 202, 0.2),
        (105, 208, 0.4),
        (110, 220, 1.0),
        (120, 240, 1.6)
    ]
    
    features = preprocess_data(sample_data)
    print("Extracted Features:", features)

