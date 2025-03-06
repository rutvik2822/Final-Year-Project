import numpy as np

def preprocess_data(drawing_data):
    """
    Preprocess drawing data to extract meaningful features for ML analysis.
    :param drawing_data: List of tuples [(x, y, timestamp), ...]
    :return: Dictionary of extracted features
    """
    speeds = []
    pauses = []
    timestamps = [point[2] for point in drawing_data]
    
    for i in range(1, len(drawing_data)):
        x1, y1, t1 = drawing_data[i - 1]
        x2, y2, t2 = drawing_data[i]
        
        # Calculate speed
        distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        time_diff = t2 - t1
        if time_diff > 0:
            speed = distance / time_diff
            speeds.append(speed)
        
        # Calculate pauses (time differences > threshold)
        if time_diff > 0.5:  # Example threshold: 0.5 seconds
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
