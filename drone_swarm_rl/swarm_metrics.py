import numpy as np
from typing import Dict, Tuple, List
from .drone import Drone

def calculate_centroid(drones: Dict[str, Drone]) -> np.ndarray:
    """
    Calculate the centroid of the swarm.
    
    Args:
        drones: Dictionary of Drone objects
        
    Returns:
        np.ndarray: 3D position of the swarm centroid
    """
    positions = np.array([drone.get_position() for drone in drones.values()])
    return np.mean(positions, axis=0)

def calculate_velocity_centroid(drones: Dict[str, Drone]) -> np.ndarray:
    """
    Calculate the average velocity of the swarm.
    
    Args:
        drones: Dictionary of Drone objects
        
    Returns:
        np.ndarray: 3D velocity vector of the swarm centroid
    """
    velocities = np.array([drone.get_velocity() for drone in drones.values()])
    return np.mean(velocities, axis=0)

def calculate_orientation_centroid(drones: Dict[str, Drone]) -> np.ndarray:
    """
    Calculate the average orientation of the swarm.
    Note: This is a simplified calculation that may not be appropriate for all cases
    due to the circular nature of angles.
    
    Args:
        drones: Dictionary of Drone objects
        
    Returns:
        np.ndarray: Average roll, pitch, yaw angles
    """
    orientations = np.array([drone.get_orientation() for drone in drones.values()])
    return np.mean(orientations, axis=0)

def calculate_distance_metrics(drones: Dict[str, Drone]) -> Dict[str, float]:
    """
    Calculate various distance-based metrics for the swarm.
    
    Args:
        drones: Dictionary of Drone objects
        
    Returns:
        Dict containing:
            - avg_distance_to_centroid: Average distance of drones to centroid
            - max_distance_to_centroid: Maximum distance of any drone to centroid
            - min_distance_to_centroid: Minimum distance of any drone to centroid
            - std_distance_to_centroid: Standard deviation of distances to centroid
    """
    centroid = calculate_centroid(drones)
    positions = np.array([drone.get_position() for drone in drones.values()])
    
    distances = np.linalg.norm(positions - centroid, axis=1)
    
    return {
        'avg_distance_to_centroid': np.mean(distances),
        'max_distance_to_centroid': np.max(distances),
        'min_distance_to_centroid': np.min(distances),
        'std_distance_to_centroid': np.std(distances)
    }

def calculate_velocity_metrics(drones: Dict[str, Drone]) -> Dict[str, float]:
    """
    Calculate velocity-based metrics for the swarm.
    
    Args:
        drones: Dictionary of Drone objects
        
    Returns:
        Dict containing:
            - avg_velocity_magnitude: Average speed of drones
            - velocity_alignment: Average cosine similarity of velocity vectors
            - velocity_std: Standard deviation of velocity magnitudes
    """
    velocities = np.array([drone.get_velocity() for drone in drones.values()])
    velocity_centroid = calculate_velocity_centroid(drones)
    
    # Calculate velocity magnitudes
    velocity_magnitudes = np.linalg.norm(velocities, axis=1)
    
    # Calculate velocity alignment (cosine similarity)
    if np.linalg.norm(velocity_centroid) > 0:
        normalized_velocities = velocities / np.linalg.norm(velocities, axis=1, keepdims=True)
        normalized_centroid = velocity_centroid / np.linalg.norm(velocity_centroid)
        velocity_alignment = np.mean(np.sum(normalized_velocities * normalized_centroid, axis=1))
    else:
        velocity_alignment = 0.0
    
    return {
        'avg_velocity_magnitude': np.mean(velocity_magnitudes),
        'velocity_alignment': velocity_alignment,
        'velocity_std': np.std(velocity_magnitudes)
    }

def calculate_orientation_metrics(drones: Dict[str, Drone]) -> Dict[str, float]:
    """
    Calculate orientation-based metrics for the swarm.
    
    Args:
        drones: Dictionary of Drone objects
        
    Returns:
        Dict containing:
            - orientation_alignment: Average cosine similarity of orientation vectors
            - roll_std: Standard deviation of roll angles
            - pitch_std: Standard deviation of pitch angles
            - yaw_std: Standard deviation of yaw angles
    """
    orientations = np.array([drone.get_orientation() for drone in drones.values()])
    orientation_centroid = calculate_orientation_centroid(drones)
    
    # Calculate orientation alignment (cosine similarity)
    normalized_orientations = orientations / np.linalg.norm(orientations, axis=1, keepdims=True)
    normalized_centroid = orientation_centroid / np.linalg.norm(orientation_centroid)
    orientation_alignment = np.mean(np.sum(normalized_orientations * normalized_centroid, axis=1))
    
    return {
        'orientation_alignment': orientation_alignment,
        'roll_std': np.std(orientations[:, 0]),
        'pitch_std': np.std(orientations[:, 1]),
        'yaw_std': np.std(orientations[:, 2])
    }

def identify_outlier_drones(drones: Dict[str, Drone], 
                          distance_threshold: float = 50.0,
                          velocity_threshold: float = 0.5,
                          orientation_threshold: float = 0.5) -> Dict[str, List[str]]:
    """
    Identify drones that are outliers based on position, velocity, and orientation.
    
    Args:
        drones: Dictionary of Drone objects
        distance_threshold: Maximum allowed distance from centroid
        velocity_threshold: Minimum required velocity alignment
        orientation_threshold: Minimum required orientation alignment
        
    Returns:
        Dict containing lists of outlier drone IDs for each metric
    """
    centroid = calculate_centroid(drones)
    velocity_centroid = calculate_velocity_centroid(drones)
    orientation_centroid = calculate_orientation_centroid(drones)
    
    outliers = {
        'distance_outliers': [],
        'velocity_outliers': [],
        'orientation_outliers': []
    }
    
    for drone_id, drone in drones.items():
        # Check distance
        distance = np.linalg.norm(drone.get_position() - centroid)
        if distance > distance_threshold:
            outliers['distance_outliers'].append(drone_id)
        
        # Check velocity alignment
        velocity = drone.get_velocity()
        if np.linalg.norm(velocity) > 0 and np.linalg.norm(velocity_centroid) > 0:
            velocity_alignment = np.dot(velocity, velocity_centroid) / (
                np.linalg.norm(velocity) * np.linalg.norm(velocity_centroid))
            if velocity_alignment < velocity_threshold:
                outliers['velocity_outliers'].append(drone_id)
        
        # Check orientation alignment
        orientation = drone.get_orientation()
        orientation_alignment = np.dot(orientation, orientation_centroid) / (
            np.linalg.norm(orientation) * np.linalg.norm(orientation_centroid))
        if orientation_alignment < orientation_threshold:
            outliers['orientation_outliers'].append(drone_id)
    
    return outliers

def calculate_swarm_health(drones: Dict[str, Drone],
                         distance_threshold: float = 50.0,
                         velocity_threshold: float = 0.5,
                         orientation_threshold: float = 0.5) -> float:
    """
    Calculate an overall health score for the swarm (0.0 to 1.0).
    
    Args:
        drones: Dictionary of Drone objects
        distance_threshold: Maximum allowed distance from centroid
        velocity_threshold: Minimum required velocity alignment
        orientation_threshold: Minimum required orientation alignment
        
    Returns:
        float: Swarm health score between 0.0 and 1.0
    """
    # Get all metrics
    distance_metrics = calculate_distance_metrics(drones)
    velocity_metrics = calculate_velocity_metrics(drones)
    orientation_metrics = calculate_orientation_metrics(drones)
    
    # Calculate individual scores
    distance_score = 1.0 - min(1.0, distance_metrics['max_distance_to_centroid'] / distance_threshold)
    velocity_score = velocity_metrics['velocity_alignment']
    orientation_score = orientation_metrics['orientation_alignment']
    
    # Combine scores with weights
    weights = {
        'distance': 0.4,
        'velocity': 0.3,
        'orientation': 0.3
    }
    
    health_score = (
        weights['distance'] * distance_score +
        weights['velocity'] * velocity_score +
        weights['orientation'] * orientation_score
    )
    
    return health_score 