import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

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
    
    def __init__(self, num_drones=3, max_steps=1000):
        super().__init__()
        
        self.num_drones = num_drones
        self.max_steps = max_steps
        self.current_step = 0
        
        # Physical parameters
        self.gravity = 9.81  # m/s^2
        self.mass = 0.5  # kg
        self.max_thrust = 15.0  # N
        self.max_angular_vel = 2.0  # rad/s
        
        # Visualization setup
        self.fig = None
        self.ax = None
        self.drone_trails = {f'drone_{i}': [] for i in range(num_drones)}
        self.trail_length = 50  # Number of positions to keep in trail
        
        # Define observation space for each drone
        # [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        single_drone_obs_space = spaces.Box(
            low=np.array([
                -100, -100, 0,      # Position bounds (x,y,z)
                -20, -20, -20,      # Velocity bounds (vx,vy,vz)
                -np.pi, -np.pi, -np.pi,  # Orientation bounds (roll,pitch,yaw)
                -10, -10, -10       # Angular velocity bounds (wx,wy,wz)
            ]),
            high=np.array([
                100, 100, 100,      # Position bounds
                20, 20, 20,         # Velocity bounds
                np.pi, np.pi, np.pi,  # Orientation bounds
                10, 10, 10          # Angular velocity bounds
            ]),
            dtype=np.float32
        )
        
        # Define action space for each drone
        # [thrust, roll_rate, pitch_rate, yaw_rate]
        single_drone_action_space = spaces.Box(
            low=np.array([0, -1, -1, -1]),
            high=np.array([1, 1, 1, 1]),
            dtype=np.float32
        )
        
        # Combine spaces for all drones
        self.observation_space = gym.spaces.Dict({
            f'drone_{i}': single_drone_obs_space for i in range(num_drones)
        })
        
        self.action_space = gym.spaces.Dict({
            f'drone_{i}': single_drone_action_space for i in range(num_drones)
        })
        
        # Initialize state
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Initialize each drone with random position and zero velocity/orientation
        self.state = {}
        for i in range(self.num_drones):
            self.state[f'drone_{i}'] = np.zeros(12)
            # Random initial position in a 10x10x10 cube
            self.state[f'drone_{i}'][:3] = self.np_random.uniform(low=-5, high=5, size=3)
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        return {
            drone_id: self.state[drone_id].astype(np.float32)
            for drone_id in self.state
        }
    
    def step(self, action):
        self.current_step += 1
        
        # Update state for each drone based on actions
        for drone_id, drone_action in action.items():
            self._update_drone_state(drone_id, drone_action)
        
        # Calculate reward
        reward = self._compute_reward()
        
        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {}
    
    def _update_drone_state(self, drone_id, action):
        # Extract current state
        state = self.state[drone_id]
        pos = state[0:3]
        vel = state[3:6]
        orientation = state[6:9]
        angular_vel = state[9:12]
        
        # Extract actions
        thrust = action[0] * self.max_thrust
        angular_rates = action[1:4] * self.max_angular_vel
        
        # Simple physics update (Euler integration)
        dt = 0.05  # 50ms timestep
        
        # Update position and velocity
        acceleration = np.array([0, 0, -self.gravity])  # gravity
        acceleration[2] += thrust / self.mass  # thrust in z-direction
        
        vel += acceleration * dt
        pos += vel * dt
        
        # Update orientation and angular velocity
        orientation += angular_vel * dt
        angular_vel = angular_rates  # Direct control of angular velocity
        
        # Update state
        self.state[drone_id] = np.concatenate([pos, vel, orientation, angular_vel])
    
    def _compute_reward(self):
        # Simple reward function based on maintaining formation
        # You can modify this based on your specific requirements
        reward = 0
        
        # Calculate centroid
        positions = np.array([self.state[f'drone_{i}'][:3] for i in range(self.num_drones)])
        centroid = positions.mean(axis=0)
        
        # Reward based on distance to centroid
        for i in range(self.num_drones):
            dist_to_centroid = np.linalg.norm(positions[i] - centroid)
            reward -= dist_to_centroid
        
        return reward
    
    def render(self):
        """
        Render the environment with matplotlib.
        Shows:
        - Drone positions as points
        - Drone orientations as arrows
        - Trails showing recent positions
        - Swarm centroid
        """
        if self.fig is None or self.ax is None:
            self.fig = plt.figure(figsize=(10, 10))
            self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.cla()  # Clear the current axes
        
        # Set axis labels and limits
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim([-100, 100])
        self.ax.set_ylim([-100, 100])
        self.ax.set_zlim([0, 100])
        
        # Get positions of all drones
        positions = []
        orientations = []
        for i in range(self.num_drones):
            drone_id = f'drone_{i}'
            state = self.state[drone_id]
            pos = state[0:3]
            orientation = state[6:9]  # roll, pitch, yaw
            
            positions.append(pos)
            orientations.append(orientation)
            
            # Update and plot trail
            self.drone_trails[drone_id].append(pos.copy())
            if len(self.drone_trails[drone_id]) > self.trail_length:
                self.drone_trails[drone_id].pop(0)
            
            trail = np.array(self.drone_trails[drone_id])
            if len(trail) > 1:
                self.ax.plot3D(trail[:, 0], trail[:, 1], trail[:, 2], 
                             alpha=0.3, linestyle='--', color=f'C{i}')
        
        positions = np.array(positions)
        
        # Plot drones as points
        self.ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                       c=range(self.num_drones), cmap='viridis', 
                       s=100, marker='o')
        
        # Plot orientation arrows
        arrow_length = 5.0
        for i, (pos, orient) in enumerate(zip(positions, orientations)):
            # Simplified orientation visualization - just showing yaw
            yaw = orient[2]
            dx = arrow_length * np.cos(yaw)
            dy = arrow_length * np.sin(yaw)
            self.ax.quiver(pos[0], pos[1], pos[2], 
                         dx, dy, 0,
                         color=f'C{i}', 
                         arrow_length_ratio=0.2)
        
        # Plot centroid
        centroid = positions.mean(axis=0)
        self.ax.scatter([centroid[0]], [centroid[1]], [centroid[2]], 
                       color='red', marker='*', s=200, label='Centroid')
        
        # Add legend
        self.ax.legend()
        
        # Add title with step information
        self.ax.set_title(f'Step {self.current_step}')
        
        # Draw and pause briefly
        plt.draw()
        plt.pause(0.01) 