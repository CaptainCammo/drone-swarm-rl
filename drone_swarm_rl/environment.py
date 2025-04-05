import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .drone import Drone
from .swarm_metrics import calculate_swarm_health, calculate_distance_metrics, calculate_velocity_metrics, calculate_orientation_metrics, identify_outlier_drones

class DroneSwarmEnv(gym.Env):
    """
    A 3D environment for controlling multiple drones in a swarm.
    
    State space (per drone):
        - Position (x, y, z)
        - Velocity (vx, vy, vz)
        - Orientation (roll, pitch, yaw)
        - Angular velocity (wx, wy, wz)
    
    Action space (per drone):
        - Thrust (normalized between 0 and 1)
        - Roll rate command
        - Pitch rate command
        - Yaw rate command
    """
    
    def __init__(self, num_drones=3, max_steps=80):
        super().__init__()
        
        self.num_drones = num_drones
        self.max_steps = max_steps
        self.current_step = 0
        
        # Initialize drones
        self.drones = {}
        for i in range(num_drones):
            # Initialize drones in a line formation
            initial_position = np.array([i * 10.0, 0.0, 0.0])
            self.drones[f'drone_{i}'] = Drone(f'drone_{i}', initial_position=initial_position)
        
        # Define action space
        self.action_space = spaces.Dict({
            f'drone_{i}': spaces.Box(
                low=np.array([0, -1, -1, -1], dtype=np.float32),
                high=np.array([1, 1, 1, 1], dtype=np.float32),
                dtype=np.float32
            ) for i in range(num_drones)
        })
        
        # Define observation space
        self.observation_space = spaces.Dict({
            f'drone_{i}': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(12,),  # [position(3), velocity(3), orientation(3), angular_velocity(3)]
                dtype=np.float32
            ) for i in range(num_drones)
        })
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset step counter
        self.current_step = 0
        
        # Define the target centroid position
        target_centroid = np.array([-40, 0, 50])
        
        # Initialize each drone with position around the target centroid
        positions = []
        
        # First, generate random positions in a small cube
        for i in range(self.num_drones):
            # Random position in a 10x10x10 cube centered at origin
            pos = self.np_random.uniform(low=-5, high=5, size=3)
            positions.append(pos)
        
        # Calculate the current centroid of these positions
        current_centroid = np.mean(positions, axis=0)
        
        # Calculate the offset needed to move the centroid to the target
        offset = target_centroid - current_centroid
        
        # Apply the offset to all positions and initialize drones
        for i in range(self.num_drones):
            # Set initial position with offset applied
            initial_position = positions[i] + offset
            
            # Set initial velocity for level flight (along x-axis)
            # This gives enough airspeed to generate lift
            initial_velocity = np.array([10.0, 0, 0])  # 10 m/s forward
            
            # Set initial orientation for level flight
            # Yaw = 0 (facing positive x-axis), slight negative pitch to maintain altitude
            initial_orientation = np.array([0, +0.05, 0])  # [roll, pitch, yaw]
            
            # Initialize drone with specified state
            self.drones[f'drone_{i}'] = Drone(
                f'drone_{i}',
                initial_position=initial_position,
                initial_velocity=initial_velocity,
                initial_orientation=initial_orientation
            )
        
        return self._get_obs(), {}
    
    def step(self, action):
        self.current_step += 1
        
        # Update state for each drone based on actions
        for drone_id, drone_action in action.items():
            self.drones[drone_id].update_state(drone_action)
        
        # Calculate positions and centroid
        positions = np.array([drone.get_position() for drone in self.drones.values()])
        centroid = positions.mean(axis=0)
        
        # Calculate reward
        reward = self._compute_reward(centroid, positions)
        
        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        
        # Check if centroid is outside the boundary
        boundary_x = [-300, 300]
        boundary_y = [-300, 300]
        boundary_z = [0, 300]
        
        # Initialize info dictionary
        info = {}
        
        # Terminate if centroid is outside the boundary
        if (centroid[0] < boundary_x[0] or centroid[0] > boundary_x[1] or
            centroid[1] < boundary_y[0] or centroid[1] > boundary_y[1] or
            centroid[2] < boundary_z[0] or centroid[2] > boundary_z[1]):
            terminated = True
            reward -= 100  # Large penalty for going out of bounds
            
            # Add termination reason to info
            if centroid[0] < boundary_x[0]:
                info['terminated_reason'] = 'out_of_bounds_x_min'
            elif centroid[0] > boundary_x[1]:
                info['terminated_reason'] = 'out_of_bounds_x_max'
            elif centroid[1] < boundary_y[0]:
                info['terminated_reason'] = 'out_of_bounds_y_min'
            elif centroid[1] > boundary_y[1]:
                info['terminated_reason'] = 'out_of_bounds_y_max'
            elif centroid[2] < boundary_z[0]:
                info['terminated_reason'] = 'out_of_bounds_z_min'
            elif centroid[2] > boundary_z[1]:
                info['terminated_reason'] = 'out_of_bounds_z_max'
        
        # Add max steps termination reason
        if self.current_step >= self.max_steps:
            info['terminated_reason'] = 'max_steps'
        
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self):
        """Get the current observation."""
        return {drone_id: drone.get_state() for drone_id, drone in self.drones.items()}
    
    def _compute_reward(self, centroid, positions):
        """Compute the reward based on the current state."""
        # Calculate swarm health score
        health_score = calculate_swarm_health(self.drones)
        
        # Get distance metrics for more granular penalties
        distance_metrics = calculate_distance_metrics(self.drones)
        
        # Identify outlier drones
        outliers = identify_outlier_drones(self.drones, 
                                         distance_threshold=50.0,
                                         velocity_threshold=0.5,
                                         orientation_threshold=0.5)
        
        # Base reward from swarm health
        reward = health_score * 20.0 
        
        # Severe penalties for outlier drones
        num_outliers = len(outliers['distance_outliers'])
        if num_outliers> 0:
            # Penalty increases with number of outlier drones
            reward -= 10.0 * num_outliers
        
        # Additional penalties for poor formation
        if distance_metrics['max_distance_to_centroid'] > 50.0:
            # Quadratic penalty for maximum distance
            reward -= 0.1 * (distance_metrics['max_distance_to_centroid'] - 50.0) ** 2
        
        if distance_metrics['std_distance_to_centroid'] > 20.0:
            # Quadratic penalty for high standard deviation
            reward -= 0.1 * (distance_metrics['std_distance_to_centroid'] - 20.0) ** 2
        
        return reward
    
    def render(self, return_fig=False):
        """Render the environment."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        if not hasattr(self, 'fig'):
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.clear()
        
        # Plot drones
        for drone in self.drones.values():
            pos = drone.get_position()
            self.ax.scatter(pos[0], pos[1], pos[2], c='b', marker='o')
        
        # Set plot limits
        self.ax.set_xlim([-300, 300])
        self.ax.set_ylim([-300, 300])
        self.ax.set_zlim([0, 300])
        
        # Add labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        # Add title
        self.ax.set_title(f"Step {self.current_step}")
        
        if return_fig:
            return self.fig, self.ax
        
        plt.draw()
        plt.pause(0.01) 