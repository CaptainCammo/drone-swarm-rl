import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .environment import DroneSwarmEnv
from .swarm_metrics import calculate_centroid

class WaypointDroneEnv(DroneSwarmEnv):
    """
    A drone swarm environment with waypoints (3D boxes) that the drones need to navigate through.
    The agent is rewarded for guiding the swarm's centroid through the waypoints in order.
    """
    
    def __init__(self, num_drones=3, max_steps=1000, num_waypoints=5, 
                 waypoint_size=15.0, random_waypoints=True, waypoint_path=None,
                 waypoint_reward=1000.0, distance_reward_factor=0.05, reward_exponent=1.5,
                 max_steps_away=50, wrong_direction_penalty=200.0):
        """
        Initialize the waypoint environment.
        
        Args:
            num_drones: Number of drones in the swarm
            max_steps: Maximum steps per episode
            num_waypoints: Number of waypoints to generate
            waypoint_size: Size of the waypoint boxes
            random_waypoints: Whether to generate random waypoints
            waypoint_path: List of waypoint coordinates (if not random)
            waypoint_reward: Reward for reaching a waypoint
            distance_reward_factor: Factor for distance-based reward component
            reward_exponent: Exponent for scaling waypoint rewards (higher = more exponential growth)
            max_steps_away: Maximum consecutive steps allowed moving away from waypoint
            wrong_direction_penalty: Penalty for moving away from waypoint for too long
        """
        # Set waypoint attributes before calling parent's __init__
        self.num_waypoints = num_waypoints
        self.waypoint_size = waypoint_size
        self.random_waypoints = random_waypoints
        self.waypoint_path = waypoint_path
        self.waypoint_reward = waypoint_reward
        self.distance_reward_factor = distance_reward_factor
        self.reward_exponent = reward_exponent
        self.max_steps_away = max_steps_away
        self.wrong_direction_penalty = wrong_direction_penalty
        
        # Initialize waypoint tracking variables
        self.waypoints = []
        self.waypoint_reached = []
        self.current_waypoint_index = 0
        self.prev_distance_to_waypoint = None
        self.steps_moving_away = 0  # Counter for consecutive steps moving away from waypoint
        
        # Generate colors for waypoints
        self.waypoint_colors = self._generate_waypoint_colors()
        
        # Tell parent class to skip its reset during initialization
        # We'll handle the reset ourselves after initialization
        self._skip_reset_in_init = True
        
        # Call parent's __init__
        super().__init__(num_drones=num_drones, max_steps=max_steps)
        
        # Reset the skip flag
        self._skip_reset_in_init = False
        
        # Now manually call reset to initialize the environment
        self.reset()
        
    def _generate_waypoint_colors(self):
        """Generate distinct colors for waypoints."""
        import matplotlib.cm as cm
        
        # Use a colormap to generate distinct colors
        cmap = cm.get_cmap('viridis', self.num_waypoints)
        colors = [cmap(i) for i in range(self.num_waypoints)]
        return colors
    
    def _generate_waypoints(self):
        """Generate waypoints for the environment."""
        if not self.random_waypoints and self.waypoint_path:
            # Use provided waypoint path
            self.waypoints = self.waypoint_path[:self.num_waypoints]
        else:
            # Generate random waypoints within the expanded space
            self.waypoints = []
            for _ in range(self.num_waypoints):
                # Generate waypoint within the expanded range
                x = np.random.uniform(-250, 250)
                y = np.random.uniform(-250, 250)
                z = np.random.uniform(30, 250)  # Keep waypoints above ground
                self.waypoints.append(np.array([x, y, z]))
        
        # Reset waypoint tracking
        self.waypoint_reached = [False] * len(self.waypoints)
        self.current_waypoint_index = 0
    
    def reset(self, seed=None, options=None):
        """Reset the environment and generate new waypoints."""
        # Reset the base environment
        obs, info = super().reset(seed=seed, options=options)
        
        # Generate waypoints
        self._generate_waypoints()
        
        # Initialize distance to current waypoint
        self.prev_distance_to_waypoint = self._get_distance_to_current_waypoint()
        
        # Reset steps moving away counter
        self.steps_moving_away = 0
        
        # Add waypoint info to info dict
        info['current_waypoint_index'] = self.current_waypoint_index
        info['waypoints_reached'] = 0
        info['all_waypoints_reached'] = False
        
        return obs, info
    
    def _get_distance_to_current_waypoint(self):
        """Calculate distance from swarm centroid to current waypoint."""
        if self.current_waypoint_index >= self.num_waypoints:
            return 0.0
        
        # Get positions and centroid using swarm_metrics
        positions = np.array([drone.get_position() for drone in self.drones.values()])
        centroid = calculate_centroid(self.drones)
        
        # Calculate distance to current waypoint
        current_waypoint = self.waypoints[self.current_waypoint_index]
        distance = np.linalg.norm(centroid - current_waypoint)
        
        return distance
    
    def _check_waypoint_reached(self):
        """Check if the current waypoint has been reached."""
        if self.current_waypoint_index >= len(self.waypoints):
            return False
        
        # Get positions and centroid using swarm_metrics
        positions = np.array([drone.get_position() for drone in self.drones.values()])
        centroid = calculate_centroid(self.drones)
        
        # Check if centroid is within the waypoint box
        current_waypoint = self.waypoints[self.current_waypoint_index]
        half_size = self.waypoint_size / 2
        
        # Check if centroid is within the waypoint box
        within_x = abs(centroid[0] - current_waypoint[0]) < half_size
        within_y = abs(centroid[1] - current_waypoint[1]) < half_size
        within_z = abs(centroid[2] - current_waypoint[2]) < half_size
        
        return within_x and within_y and within_z
    
    def step(self, action):
        """Take a step in the environment and check waypoints."""
        # Get the base environment step result
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Get positions and centroid using swarm_metrics
        positions = np.array([drone.get_position() for drone in self.drones.values()])
        centroid = calculate_centroid(self.drones)
        
        # Only calculate waypoint-related rewards if we haven't reached all waypoints
        if self.current_waypoint_index < len(self.waypoints):
            # Calculate distance to current waypoint
            current_waypoint = self.waypoints[self.current_waypoint_index]
            distance_to_waypoint = np.linalg.norm(centroid - current_waypoint)
            
            # Check if we've reached the current waypoint
            if distance_to_waypoint < self.waypoint_size:
                # Add waypoint reward
                reward += self.waypoint_reward
                info['waypoint_reward'] = self.waypoint_reward
                
                # Move to next waypoint
                self.current_waypoint_index += 1
                
                # Reset steps moving away counter
                self.steps_moving_away = 0
                
                # Check if we've reached all waypoints
                if self.current_waypoint_index >= len(self.waypoints):
                    terminated = True
                    info['terminated_reason'] = 'all_waypoints_reached'
                    reward += self.waypoint_reward * 2  # Bonus for completing all waypoints
            else:
                # Calculate if we're moving toward or away from the waypoint
                if self.prev_distance_to_waypoint is not None:
                    if distance_to_waypoint >= self.prev_distance_to_waypoint:
                        # Moving away from waypoint
                        self.steps_moving_away += 1
                        
                        # Apply penalty if we've been moving away for too long
                        if self.steps_moving_away >= self.max_steps_away:
                            terminated = True
                            reward -= self.wrong_direction_penalty
                            info['wrong_direction_penalty'] = self.wrong_direction_penalty
                            info['terminated_reason'] = 'moving_away_too_long'
                    else:
                        # Moving toward waypoint
                        self.steps_moving_away = 0
                
                # Add distance-based reward
                distance_reward = self.distance_reward_factor * (1.0 / (1.0 + distance_to_waypoint))
                reward += distance_reward
                info['distance_reward'] = distance_reward
            
            # Update previous distance
            self.prev_distance_to_waypoint = distance_to_waypoint
        else:
            # If we've reached all waypoints, set the distance to 0
            self.prev_distance_to_waypoint = 0.0
        
        # Add waypoint info
        info['current_waypoint_index'] = self.current_waypoint_index
        info['distance_to_waypoint'] = self.prev_distance_to_waypoint
        info['steps_moving_away'] = self.steps_moving_away
        
        # Check if centroid is outside the boundary
        boundary_x = [-300, 300]
        boundary_y = [-300, 300]
        boundary_z = [0, 300]
        
        if (centroid[0] < boundary_x[0] or centroid[0] > boundary_x[1] or
            centroid[1] < boundary_y[0] or centroid[1] > boundary_y[1] or
            centroid[2] < boundary_z[0] or centroid[2] > boundary_z[1]):
            terminated = True
            reward -= 100  # Large penalty for going out of bounds
            
            # Add specific out-of-bounds reason
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
        if self.current_step >= self.max_steps and not terminated:
            info['terminated_reason'] = 'max_steps'
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self, centroid, positions):
        """Compute the reward based on the current state."""
        from .swarm_metrics import calculate_swarm_health
        
        # Calculate swarm health score
        health_score = calculate_swarm_health(self.drones)
        
        # Base reward from swarm health
        reward = health_score * 10.0
        
        # Add waypoint-specific rewards
        if self.current_waypoint_index < len(self.waypoints):
            current_waypoint = self.waypoints[self.current_waypoint_index]
            distance_to_waypoint = np.linalg.norm(centroid - current_waypoint)
            
            # Check if any drone is out of bounds
            boundary_x = [-300, 300]
            boundary_y = [-300, 300]
            boundary_z = [0, 300]
            
            out_of_bounds = False
            for drone in self.drones.values():
                pos = drone.get_position()
                if (pos[0] < boundary_x[0] or pos[0] > boundary_x[1] or
                    pos[1] < boundary_y[0] or pos[1] > boundary_y[1] or
                    pos[2] < boundary_z[0] or pos[2] > boundary_z[1]):
                    out_of_bounds = True
                    break
            
            # If out of bounds before reaching waypoint, apply strong penalty
            if out_of_bounds:
                reward -= (100.0 * self.num_waypoints / (self.current_waypoint_index + 1)) # Strong penalty for going out of bounds before reaching waypoint
            elif distance_to_waypoint < self.waypoint_size:
                reward += self.waypoint_reward
                self.current_waypoint_index += 1
            else:
                # Exponential reward for getting closer to the waypoint
                reward += self.distance_reward_factor * np.exp(-distance_to_waypoint / self.waypoint_size)
        
        return reward
    
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
            alpha = 0.8 if i == self.current_waypoint_index else 0.3
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
        self.ax.set_title(f"Step {self.current_step} - Current Waypoint: {self.current_waypoint_index + 1}/{len(self.waypoints)}")
        
        if return_fig:
            return fig, self.ax
        else:
            # Update the display
            plt.draw()
            plt.pause(0.01)
            return None

def create_scanning_waypoints():
    """
    Create a set of 10 waypoints that form a scanning pattern through the expanded 3D space.
    The pattern is designed with much wider spacing and gentler turns to make it easier
    for drones to navigate, with long straight segments between turns.
    """
    # Starting point - adjusted for the expanded space
    start_x, start_y, start_z = -40, 0, 50
    
    # Define a much more spread out scanning pattern with gentler turns
    waypoints = [
        # First long straight segment
        np.array([start_x + 100, start_y, start_z]),
        np.array([start_x + 200, start_y, start_z + 100]),
        
        # Wide turn and second straight segment (shifted north)
        np.array([start_x + 400, start_y + 150, start_z]),
        np.array([start_x, start_y + 150, start_z]),
        
        # Wide turn and third straight segment (shifted north again)
        np.array([start_x, start_y + 300, start_z]),
        np.array([start_x + 400, start_y + 300, start_z]),
        
        # Climb to higher altitude with a gentle slope
        np.array([start_x + 400, start_y + 300, start_z + 100]),
        
        # Fourth straight segment at higher altitude
        np.array([start_x, start_y + 300, start_z + 100]),
        
        # Final straight segment completing the pattern
        np.array([start_x, start_y, start_z + 100]),
        np.array([start_x + 400, start_y, start_z + 100])
    ]
    
    return waypoints