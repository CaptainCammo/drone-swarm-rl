import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .environment import DroneSwarmEnv

class WaypointDroneEnv(DroneSwarmEnv):
    """
    A drone swarm environment with waypoints (3D boxes) that the drones need to navigate through.
    The agent is rewarded for guiding the swarm's centroid through the waypoints in order.
    """
    
    def __init__(self, num_drones=3, max_steps=1000, num_waypoints=5, waypoint_size=10.0, random_waypoints=True, waypoint_path=None):
        """
        Initialize the waypoint environment.
        
        Args:
            num_drones: Number of drones in the swarm
            max_steps: Maximum number of steps per episode
            num_waypoints: Number of waypoint boxes to generate
            waypoint_size: Size of each waypoint box (cube side length)
            random_waypoints: Whether to generate random waypoints or use a predefined path
            waypoint_path: List of (x, y, z) coordinates for waypoints if random_waypoints is False
        """
        # Initialize attributes before calling parent's __init__
        self.num_waypoints = num_waypoints
        self.waypoint_size = waypoint_size
        self.random_waypoints = random_waypoints
        self.predefined_waypoint_path = waypoint_path
        
        # Waypoint tracking
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.waypoint_reached = []
        
        # Visualization properties
        self.waypoint_colors = plt.cm.viridis(np.linspace(0, 1, num_waypoints))
        
        # Call parent's __init__ but prevent it from calling reset
        self._skip_reset_in_init = True
        super().__init__(num_drones=num_drones, max_steps=max_steps)
        self._skip_reset_in_init = False
        
        # Now manually call reset to initialize the environment
        self.reset()
        
    def reset(self, seed=None, options=None):
        """Reset the environment and generate new waypoints."""
        obs, info = super().reset(seed=seed, options=options)
        
        # Generate waypoints
        self._generate_waypoints()
        
        # Reset waypoint tracking
        self.current_waypoint_idx = 0
        self.waypoint_reached = [False] * self.num_waypoints
        
        return obs, info
    
    def _generate_waypoints(self):
        """Generate waypoints either randomly or using the predefined path."""
        self.waypoints = []
        
        if self.random_waypoints:
            # Start with a waypoint near the initial drone position
            initial_positions = np.array([self.state[f'drone_{i}'][:3] for i in range(self.num_drones)])
            initial_centroid = initial_positions.mean(axis=0)
            
            # First waypoint is ahead of the drones in their initial direction
            first_waypoint = initial_centroid + np.array([30, 0, 0])  # 30 units ahead in x-direction
            self.waypoints.append(first_waypoint)
            
            # Generate subsequent waypoints with reasonable spacing
            prev_waypoint = first_waypoint
            for i in range(1, self.num_waypoints):
                # Random direction but with some continuity from previous waypoint
                direction = self.np_random.uniform(-1, 1, size=3)
                direction = direction / np.linalg.norm(direction)  # Normalize
                
                # Distance between waypoints (30-50 units)
                distance = self.np_random.uniform(30, 50)
                
                # New waypoint position
                new_waypoint = prev_waypoint + direction * distance
                
                # Ensure waypoint is within reasonable bounds
                new_waypoint = np.clip(new_waypoint, [-80, -80, 20], [80, 80, 80])
                
                self.waypoints.append(new_waypoint)
                prev_waypoint = new_waypoint
        else:
            # Use predefined waypoint path if provided
            if self.predefined_waypoint_path is not None:
                self.waypoints = [np.array(wp) for wp in self.predefined_waypoint_path[:self.num_waypoints]]
            else:
                # Default path if none provided - ensure these are numpy arrays
                self.waypoints = [
                    np.array([30.0, 0.0, 50.0]),    # Ahead
                    np.array([60.0, 30.0, 50.0]),   # Right turn
                    np.array([60.0, 60.0, 50.0]),   # Continue right
                    np.array([30.0, 60.0, 50.0]),   # Left turn
                    np.array([0.0, 30.0, 50.0])     # Return toward start
                ][:self.num_waypoints]
        
        # Debug output
        print(f"Generated {len(self.waypoints)} waypoints:")
        for i, wp in enumerate(self.waypoints):
            print(f"  Waypoint {i+1}: {wp}, type: {type(wp)}")
    
    def step(self, action):
        """
        Step the environment and check if waypoints have been reached.
        Modifies the reward based on waypoint progress.
        """
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Calculate centroid position
        positions = np.array([self.state[f'drone_{i}'][:3] for i in range(self.num_drones)])
        centroid = positions.mean(axis=0)
        
        # Check if current waypoint is reached
        waypoint_reward = self._check_waypoint_reached(centroid)
        
        # Add waypoint information to info dictionary
        info['current_waypoint'] = self.current_waypoint_idx
        info['waypoints_reached'] = sum(self.waypoint_reached)
        info['waypoint_positions'] = self.waypoints
        
        # Check if all waypoints are reached
        if all(self.waypoint_reached):
            info['all_waypoints_reached'] = True
            waypoint_reward += 100  # Bonus for completing all waypoints
            terminated = True
        else:
            info['all_waypoints_reached'] = False
        
        # Combine rewards
        total_reward = reward + waypoint_reward
        
        return obs, total_reward, terminated, truncated, info
    
    def _check_waypoint_reached(self, centroid):
        """
        Check if the current waypoint has been reached by the centroid.
        Returns a reward if a waypoint is reached.
        """
        if self.current_waypoint_idx >= len(self.waypoints):
            return 0  # All waypoints already processed
            
        # Current waypoint position
        waypoint_pos = self.waypoints[self.current_waypoint_idx]
        
        # Check if centroid is within the waypoint box
        half_size = self.waypoint_size / 2
        in_x_bounds = waypoint_pos[0] - half_size <= centroid[0] <= waypoint_pos[0] + half_size
        in_y_bounds = waypoint_pos[1] - half_size <= centroid[1] <= waypoint_pos[1] + half_size
        in_z_bounds = waypoint_pos[2] - half_size <= centroid[2] <= waypoint_pos[2] + half_size
        
        # If within bounds, mark as reached and move to next waypoint
        if in_x_bounds and in_y_bounds and in_z_bounds:
            self.waypoint_reached[self.current_waypoint_idx] = True
            reward = 50  # Reward for reaching a waypoint
            self.current_waypoint_idx += 1
            return reward
            
        # Calculate distance to current waypoint for partial reward
        distance = np.linalg.norm(centroid - waypoint_pos)
        
        # Provide a small reward for getting closer to the waypoint
        # Inversely proportional to distance, capped at a maximum value
        proximity_reward = max(0, 5 - 0.1 * distance)
        
        return proximity_reward
    
    def _compute_reward(self, centroid=None, positions=None):
        """
        Compute base reward with additional components for waypoint navigation.
        """
        # Get base reward from parent class
        base_reward = super()._compute_reward(centroid, positions)
        
        # Calculate positions and centroid if not provided
        if positions is None:
            positions = np.array([self.state[f'drone_{i}'][:3] for i in range(self.num_drones)])
        if centroid is None:
            centroid = positions.mean(axis=0)
        
        # Add waypoint direction component to encourage flying toward the next waypoint
        waypoint_direction_reward = 0
        
        if self.current_waypoint_idx < len(self.waypoints):
            # Vector from centroid to current waypoint
            to_waypoint = self.waypoints[self.current_waypoint_idx] - centroid
            distance_to_waypoint = np.linalg.norm(to_waypoint)
            
            # Normalize if not zero
            if distance_to_waypoint > 0.1:
                to_waypoint = to_waypoint / distance_to_waypoint
                
                # Get average velocity direction
                velocities = np.array([self.state[f'drone_{i}'][3:6] for i in range(self.num_drones)])
                avg_velocity = velocities.mean(axis=0)
                speed = np.linalg.norm(avg_velocity)
                
                if speed > 0.1:
                    velocity_dir = avg_velocity / speed
                    
                    # Reward for velocity aligned with direction to waypoint
                    alignment = np.dot(to_waypoint, velocity_dir)
                    waypoint_direction_reward = 2.0 * alignment
        
        # Combine rewards
        total_reward = base_reward + waypoint_direction_reward
        
        return total_reward
    
    def render(self, return_fig=False):
        """
        Render the environment with waypoints.
        
        Args:
            return_fig (bool): If True, returns the figure and axes instead of displaying
            
        Returns:
            tuple: (fig, ax) if return_fig is True, otherwise None
        """
        # Call parent class render to draw drones and get fig, ax
        result = super().render(return_fig=True)
        fig, self.ax = result
        
        # Draw waypoints as boxes
        for i, waypoint in enumerate(self.waypoints):
            # Skip rendering waypoints that have been reached
            if i < len(self.waypoint_reached) and self.waypoint_reached[i]:
                continue
                
            # Highlight the current waypoint
            alpha = 0.8 if i == self.current_waypoint_idx else 0.3
            color = self.waypoint_colors[i]
            
            # Create box vertices
            half_size = self.waypoint_size / 2
            x, y, z = waypoint
            
            # Define the 8 vertices of the cube
            vertices = np.array([
                [x - half_size, y - half_size, z - half_size],
                [x + half_size, y - half_size, z - half_size],
                [x + half_size, y + half_size, z - half_size],
                [x - half_size, y + half_size, z - half_size],
                [x - half_size, y - half_size, z + half_size],
                [x + half_size, y - half_size, z + half_size],
                [x + half_size, y + half_size, z + half_size],
                [x - half_size, y + half_size, z + half_size]
            ])
            
            # Define the 6 faces of the cube
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
                [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
                [vertices[1], vertices[2], vertices[6], vertices[5]]   # right
            ]
            
            # Create a Poly3DCollection with explicit color
            waypoint_box = Poly3DCollection(faces, alpha=alpha, linewidths=1, edgecolor='black')
            waypoint_box.set_facecolor(color)
            
            # Add the collection to the plot
            self.ax.add_collection3d(waypoint_box)
            
            # Add waypoint number with contrasting color for visibility
            text_color = 'white' if np.mean(color[:3]) < 0.5 else 'black'
            self.ax.text(x, y, z + half_size + 2, f"WP {i+1}", 
                        color=text_color, fontsize=12, ha='center', weight='bold')
        
        # Draw lines connecting waypoints to show the path
        waypoints_array = np.array(self.waypoints)
        if len(waypoints_array) > 1:
            self.ax.plot3D(waypoints_array[:, 0], waypoints_array[:, 1], waypoints_array[:, 2], 
                         'k--', alpha=0.7, linewidth=2)
            
        # Add title with step and waypoint information
        self.ax.set_title(f"Step {self.current_step} - Current Waypoint: {self.current_waypoint_idx + 1}/{len(self.waypoints)}")
        
        if return_fig:
            return fig, self.ax
        else:
            # Update the display
            plt.draw()
            plt.pause(0.01)
            return None 