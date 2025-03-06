import numpy as np
import gymnasium as gym
from gymnasium import spaces

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
        
        # Define observation space for each drone
        # [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        single_drone_obs_space = spaces.Box(
            low=np.array([-np.inf] * 12),
            high=np.array([np.inf] * 12),
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
        # Implement visualization if needed
        pass 