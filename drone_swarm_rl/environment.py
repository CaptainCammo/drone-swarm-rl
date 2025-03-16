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
        
        # Environment parameters
        self.num_drones = num_drones
        self.max_steps = max_steps
        self.current_step = 0
        
        # Physical parameters
        self.gravity = 9.81  # m/s^2
        self.mass = 0.5  # kg
        self.max_thrust = 15.0  # N
        self.max_angular_vel = 2.0  # rad/s
        
        # Airplane-specific parameters
        self.wing_area = 0.2  # m^2
        self.air_density = 1.225  # kg/m^3
        self.lift_coefficient = 2.0  # Base lift coefficient
        self.drag_coefficient = 0.1  # Base drag coefficient
        self.moment_of_inertia = np.array([0.1, 0.2, 0.15])  # kg*m^2 [roll, pitch, yaw]
        self.wing_span = 1.0  # m
        self.chord_length = 0.2  # m
        self.stall_angle = np.radians(15)  # Stall angle in radians
        
        # Define observation space for each drone
        # [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        single_drone_obs_space = gym.spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf]),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.pi, np.pi, np.pi, np.inf, np.inf, np.inf]),
            dtype=np.float32
        )
        
        # Define action space for each drone
        # [thrust, roll_rate, pitch_rate, yaw_rate]
        single_drone_action_space = gym.spaces.Box(
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
        
        # Visualization setup
        self.fig = None
        self.ax = None
        self.drone_trails = {f'drone_{i}': [] for i in range(num_drones)}
        self.trail_length = 50  # Number of positions to keep in trail
        
        # Initialize state
        self.state = {}
        
        # Skip reset during initialization if flag is set (for subclasses)
        if not hasattr(self, '_skip_reset_in_init') or not self._skip_reset_in_init:
            self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Define the target centroid position
        target_centroid = np.array([0, -90, 50])
        
        # Initialize each drone with position around the target centroid
        self.state = {}
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
            # Create state vector with zeros
            self.state[f'drone_{i}'] = np.zeros(12)
            
            # Set position with offset applied
            self.state[f'drone_{i}'][:3] = positions[i] + offset
            
            # Set initial velocity for level flight (along x-axis)
            # This gives enough airspeed to generate lift
            initial_speed = 10.0  # m/s
            self.state[f'drone_{i}'][3:6] = np.array([initial_speed, 0, 0])
            
            # Set initial orientation for level flight
            # Yaw = 0 (facing positive x-axis), slight negative pitch to maintain altitude
            self.state[f'drone_{i}'][6:9] = np.array([0, -0.05, 0])  # [roll, pitch, yaw]
        
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
        
        # Calculate positions and centroid
        positions = np.array([self.state[f'drone_{i}'][:3] for i in range(self.num_drones)])
        centroid = positions.mean(axis=0)
        
        # Calculate reward
        reward = self._compute_reward(centroid, positions)
        
        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        
        # Check if centroid is outside the boundary
        boundary_x = [-100, 100]
        boundary_y = [-100, 100]
        boundary_z = [0, 100]
        
        # Terminate if centroid is outside the boundary
        if (centroid[0] < boundary_x[0] or centroid[0] > boundary_x[1] or
            centroid[1] < boundary_y[0] or centroid[1] > boundary_y[1] or
            centroid[2] < boundary_z[0] or centroid[2] > boundary_z[1]):
            terminated = True
            reward -= 100  # Large penalty for going out of bounds
        
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, {}
    
    def _update_drone_state(self, drone_id, action):
        """Update drone state considering airplane aerodynamics"""
        # Extract current state
        state = self.state[drone_id]
        pos = state[0:3]
        vel = state[3:6]
        orientation = state[6:9]  # [roll, pitch, yaw]
        angular_vel = state[9:12]
        
        # Extract actions
        thrust = action[0] * self.max_thrust
        target_angular_rates = action[1:4] * self.max_angular_vel
        
        dt = 0.05  # 50ms timestep
        
        # Safety check for NaN values in state
        if np.any(np.isnan(pos)) or np.any(np.isnan(vel)) or np.any(np.isnan(orientation)) or np.any(np.isnan(angular_vel)):
            print(f"Warning: NaN detected in drone state. Resetting drone {drone_id} velocity and angular velocity.")
            vel = np.zeros(3)
            vel[1] = 10.0  # Reset to initial forward velocity
            angular_vel = np.zeros(3)
            # Keep position and orientation as is
        
        # Convert orientation to rotation matrix (with safety checks)
        cos_roll, cos_pitch, cos_yaw = np.cos(np.clip(orientation, -np.pi, np.pi))
        sin_roll, sin_pitch, sin_yaw = np.sin(np.clip(orientation, -np.pi, np.pi))
        
        R = np.array([
            [cos_yaw*cos_pitch, cos_yaw*sin_pitch*sin_roll - sin_yaw*cos_roll, cos_yaw*sin_pitch*cos_roll + sin_yaw*sin_roll],
            [sin_yaw*cos_pitch, sin_yaw*sin_pitch*sin_roll + cos_yaw*cos_roll, sin_yaw*sin_pitch*cos_roll - cos_yaw*sin_roll],
            [-sin_pitch, cos_pitch*sin_roll, cos_pitch*cos_roll]
        ])
        
        # Calculate airspeed and angle of attack with safety checks
        airspeed = np.linalg.norm(vel)
        
        # Ensure minimum airspeed for numerical stability
        if airspeed < 0.5:  # Increased minimum airspeed threshold
            # Apply a small boost in the current direction if speed is too low
            if airspeed > 0.1:
                vel = vel * (0.5 / airspeed)  # Scale up to minimum airspeed
            else:
                # If nearly stopped, apply velocity in the direction the drone is facing
                vel = R @ np.array([0.5, 0, 0])  # Minimum forward velocity
            airspeed = 0.5  # Update airspeed
        
        # Calculate angle of attack and sideslip with safety checks
        vel_normalized = vel / airspeed
        
        # Use safer calculation for alpha and beta with bounds
        # Forward direction in body frame is x-axis
        body_vel = R.T @ vel  # Transform velocity to body frame
        
        # Calculate alpha and beta more robustly
        if abs(body_vel[0]) > 0.1:  # If there's sufficient forward velocity
            alpha = np.arctan2(body_vel[2], body_vel[0])  # Angle of attack
        else:
            alpha = 0.0
            
        # Limit alpha to prevent extreme values
        alpha = np.clip(alpha, -np.pi/4, np.pi/4)
        
        # Calculate sideslip angle with safety
        if airspeed > 0.5:
            beta = np.arcsin(np.clip(vel_normalized[1], -0.99, 0.99))  # Sideslip angle
        else:
            beta = 0.0
            
        # Limit beta to prevent extreme values
        beta = np.clip(beta, -np.pi/4, np.pi/4)
        
        # Calculate aerodynamic forces with safety checks
        dynamic_pressure = 0.5 * self.air_density * airspeed**2
        
        # More stable lift coefficient calculation
        cl = self.lift_coefficient * np.sin(2 * alpha)  # Simplified stall model
        if abs(alpha) > self.stall_angle:
            cl *= 0.6  # Reduce lift after stall
        
        # More stable drag coefficient calculation
        cd = self.drag_coefficient * (1 + 2 * alpha**2)  # Increased drag at high AoA
        
        # Ensure coefficients are within reasonable bounds
        cl = np.clip(cl, -2.0, 2.0)
        cd = np.clip(cd, 0.05, 1.0)  # Ensure minimum drag
        
        # Calculate forces in body frame with safety checks
        lift = dynamic_pressure * self.wing_area * cl
        drag = dynamic_pressure * self.wing_area * cd
        side_force = dynamic_pressure * self.wing_area * beta * 0.1  # Simplified side force
        
        # Ensure forces are finite
        lift = np.clip(lift, -100, 100)
        drag = np.clip(drag, 0, 100)
        side_force = np.clip(side_force, -20, 20)
        
        # Combine aerodynamic forces
        aero_forces = np.array([
            -drag,
            side_force,
            -lift
        ])
        
        # Transform forces to global frame
        forces_global = R @ np.array([thrust, 0, 0]) + R @ aero_forces + np.array([0, 0, -self.mass * self.gravity])
        
        # Update velocity and position with safety checks
        acceleration = forces_global / self.mass
        acceleration = np.clip(acceleration, -20, 20)  # Limit extreme accelerations
        
        vel += acceleration * dt
        vel = np.clip(vel, -20, 20)  # Limit extreme velocities
        
        pos += vel * dt
        
        # Enforce minimum altitude
        if pos[2] < 0.1:
            pos[2] = 0.1
            vel[2] = max(0, vel[2])  # Prevent negative vertical velocity if on ground
        
        # Calculate moments with safety checks
        roll_moment = -angular_vel[0] * 0.1  # Damping
        pitch_moment = -angular_vel[1] * 0.2  # Damping
        yaw_moment = -angular_vel[2] * 0.15  # Damping
        
        # Add control surface effects with safety checks
        # Scale control effectiveness with airspeed, but ensure minimum effectiveness
        airspeed_factor = min(1.0, max(0.2, airspeed / 10.0))
        
        roll_moment += target_angular_rates[0] * airspeed_factor * 0.1
        pitch_moment += target_angular_rates[1] * airspeed_factor * 0.2
        yaw_moment += target_angular_rates[2] * airspeed_factor * 0.15
        
        # Ensure moments are finite
        roll_moment = np.clip(roll_moment, -1.0, 1.0)
        pitch_moment = np.clip(pitch_moment, -1.0, 1.0)
        yaw_moment = np.clip(yaw_moment, -1.0, 1.0)
        
        # Calculate angular acceleration
        moments = np.array([roll_moment, pitch_moment, yaw_moment])
        angular_acc = moments / self.moment_of_inertia
        angular_acc = np.clip(angular_acc, -5, 5)  # Limit extreme angular accelerations
        
        # Update angular velocity and orientation
        angular_vel += angular_acc * dt
        angular_vel = np.clip(angular_vel, -2, 2)  # Limit extreme angular velocities
        
        orientation += angular_vel * dt
        
        # Normalize angles to [-pi, pi]
        orientation = np.mod(orientation + np.pi, 2 * np.pi) - np.pi
        
        # Update state
        self.state[drone_id] = np.concatenate([pos, vel, orientation, angular_vel])
        
        # Final safety check for NaN values
        if np.any(np.isnan(self.state[drone_id])):
            print(f"Warning: NaN detected after update for drone {drone_id}. Resetting to safe values.")
            # Reset to safe values
            self.state[drone_id][3:6] = np.array([0, 10.0, 0])  # Reset velocity
            self.state[drone_id][9:12] = np.zeros(3)  # Reset angular velocity
    
    def _compute_reward(self, centroid=None, positions=None):
        """
        Compute reward based on formation maintenance and boundary constraints.
        
        Args:
            centroid: Pre-computed centroid (optional)
            positions: Pre-computed positions (optional)
        """
        # Calculate positions and centroid if not provided
        if positions is None:
            positions = np.array([self.state[f'drone_{i}'][:3] for i in range(self.num_drones)])
        if centroid is None:
            centroid = positions.mean(axis=0)
        
        # Initialize reward
        reward = 0
        
        # 1. Formation reward: negative distance to centroid
        formation_reward = 0
        for i in range(self.num_drones):
            dist_to_centroid = np.linalg.norm(positions[i] - centroid)
            formation_reward -= dist_to_centroid
        
        # 2. Boundary reward: encourage staying within safe boundaries
        # Define safe boundaries (smaller than termination boundaries)
        safe_boundary_x = [-90, 90]
        safe_boundary_y = [-90, 90]
        safe_boundary_z = [10, 90]
        
        # Calculate distance to safe boundary
        boundary_reward = 0
        
        # X-axis boundary
        if centroid[0] < safe_boundary_x[0]:
            boundary_reward -= 0.5 * (safe_boundary_x[0] - centroid[0])
        elif centroid[0] > safe_boundary_x[1]:
            boundary_reward -= 0.5 * (centroid[0] - safe_boundary_x[1])
            
        # Y-axis boundary
        if centroid[1] < safe_boundary_y[0]:
            boundary_reward -= 0.5 * (safe_boundary_y[0] - centroid[1])
        elif centroid[1] > safe_boundary_y[1]:
            boundary_reward -= 0.5 * (centroid[1] - safe_boundary_y[1])
            
        # Z-axis boundary (altitude)
        if centroid[2] < safe_boundary_z[0]:
            boundary_reward -= 1.0 * (safe_boundary_z[0] - centroid[2])  # Higher penalty for low altitude
        elif centroid[2] > safe_boundary_z[1]:
            boundary_reward -= 0.5 * (centroid[2] - safe_boundary_z[1])
        
        # 3. Combine rewards
        reward = formation_reward + boundary_reward
        
        return reward
    
    def get_stabilizing_action(self):
        """
        Generate actions that help maintain current orientation and velocity.
        Returns a dictionary of actions for each drone that will help it maintain
        stable flight without random perturbations.
        """
        actions = {}
        
        for i in range(self.num_drones):
            drone_id = f'drone_{i}'
            state = self.state[drone_id]
            
            # Extract current state
            vel = state[3:6]
            orientation = state[6:9]  # [roll, pitch, yaw]
            angular_vel = state[9:12]
            
            # Safety check for NaN values
            if np.any(np.isnan(vel)) or np.any(np.isnan(orientation)) or np.any(np.isnan(angular_vel)):
                print(f"Warning: NaN detected in drone state during action calculation. Using default action for {drone_id}.")
                actions[drone_id] = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
                continue
            
            # Calculate airspeed with safety check
            airspeed = np.linalg.norm(vel)
            
            # 1. Thrust control to maintain airspeed
            target_airspeed = 10.0  # m/s
            thrust = 0.5  # Default mid-range thrust
            
            # More aggressive thrust control for stability
            if airspeed < target_airspeed - 0.5:
                # Proportional control for thrust
                thrust_error = target_airspeed - airspeed
                thrust = 0.5 + thrust_error * 0.05  # P controller for thrust
                thrust = np.clip(thrust, 0.3, 0.9)  # Limit thrust range
            elif airspeed > target_airspeed + 0.5:
                thrust_error = airspeed - target_airspeed
                thrust = 0.5 - thrust_error * 0.05  # P controller for thrust
                thrust = np.clip(thrust, 0.3, 0.9)  # Limit thrust range
            
            # 2. Roll control to keep wings level
            # PD controller for roll
            roll_error = -orientation[0]  # Error from level (0 roll)
            roll_rate = roll_error * 0.8 - angular_vel[0] * 0.4  # Increased gains
            roll_rate = np.clip(roll_rate, -1.0, 1.0)
            
            # 3. Pitch control to maintain altitude
            # Target a slight negative pitch to generate lift
            target_pitch = -0.05
            
            # Check altitude and adjust pitch target
            pos = state[0:3]
            altitude = pos[2]
            
            # Adjust target pitch based on altitude
            if altitude < 40:  # If too low, pitch up more
                target_pitch = -0.1
            elif altitude > 60:  # If too high, pitch down more
                target_pitch = 0.0
            
            # PD controller for pitch
            pitch_error = target_pitch - orientation[1]
            pitch_rate = pitch_error * 0.8 - angular_vel[1] * 0.4  # Increased gains
            pitch_rate = np.clip(pitch_rate, -1.0, 1.0)
            
            # 4. Yaw control to maintain heading
            # Keep yaw at 0 (facing along x-axis)
            target_yaw = 0.0
            
            # Calculate yaw error using angle difference on unit circle
            yaw_error = np.arctan2(np.sin(target_yaw - orientation[2]), np.cos(target_yaw - orientation[2]))
            
            # PD controller for yaw
            yaw_rate = yaw_error * 0.8 - angular_vel[2] * 0.4  # Increased gains
            yaw_rate = np.clip(yaw_rate, -1.0, 1.0)
            
            # Combine all control inputs
            actions[drone_id] = np.array([thrust, roll_rate, pitch_rate, yaw_rate], dtype=np.float32)
        
        return actions
    
    def render(self, return_fig=False):
        """
        Render the environment with matplotlib.
        Shows:
        - Drone positions as points
        - Drone orientations as arrows
        - Trails showing recent positions
        - Swarm centroid
        
        Args:
            return_fig (bool): If True, returns the figure and axes instead of displaying
            
        Returns:
            tuple: (fig, ax) if return_fig is True, otherwise None
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
        self.ax.set_zlim([-100, 100])
        
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
        
        if return_fig:
            return self.fig, self.ax
        else:
            # Draw and pause briefly to update the display
            plt.draw()
            plt.pause(0.01)
            return None 