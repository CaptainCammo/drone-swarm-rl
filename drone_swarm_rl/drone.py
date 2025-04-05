import numpy as np

class Drone:
    def __init__(self, drone_id, initial_position=None, initial_velocity=None, initial_orientation=None):
        """
        Initialize a drone with its state space.
        
        Args:
            drone_id: Unique identifier for the drone
            initial_position: Initial position [x, y, z]
            initial_velocity: Initial velocity [vx, vy, vz]
            initial_orientation: Initial orientation [roll, pitch, yaw]
        """
        self.drone_id = drone_id
        
        # Initialize state space
        if initial_position is None:
            initial_position = np.zeros(3)
        if initial_velocity is None:
            initial_velocity = np.zeros(3)
        if initial_orientation is None:
            initial_orientation = np.zeros(3)
            
        # State space: [position(3), velocity(3), orientation(3), angular_velocity(3)]
        self.state = np.concatenate([
            initial_position,
            initial_velocity,
            initial_orientation,
            np.zeros(3)  # Initial angular velocity
        ])
        
        # Physical parameters
        self.mass = 1.0  # kg
        self.max_thrust = 20.0  # N
        self.max_angular_vel = 2.0  # rad/s
        
        # Aerodynamic parameters
        self.wing_area = 0.1  # m²
        self.air_density = 1.225  # kg/m³
        self.lift_coefficient = 0.5
        self.drag_coefficient = 0.1
        self.moment_of_inertia = np.array([0.1, 0.1, 0.1])  # kg·m²
        self.wing_span = 0.5  # m
        self.chord_length = 0.2  # m
        self.stall_angle = np.radians(15)  # radians
        
    def get_state(self):
        """Return the current state of the drone."""
        return self.state.copy()
    
    def get_position(self):
        """Return the current position of the drone."""
        return self.state[:3]
    
    def get_velocity(self):
        """Return the current velocity of the drone."""
        return self.state[3:6]
    
    def get_orientation(self):
        """Return the current orientation of the drone."""
        return self.state[6:9]
    
    def get_angular_velocity(self):
        """Return the current angular velocity of the drone."""
        return self.state[9:12]
    
    def update_state(self, action, dt=0.1):
        """
        Update the drone's state based on the action and time step.
        
        Args:
            action: [thrust, roll_rate, pitch_rate, yaw_rate]
            dt: Time step in seconds
        """
        # Unpack action
        thrust, roll_rate, pitch_rate, yaw_rate = action
        
        # Clip actions to their bounds
        thrust = np.clip(thrust, 0, 1) * self.max_thrust
        roll_rate = np.clip(roll_rate, -1, 1) * self.max_angular_vel
        pitch_rate = np.clip(pitch_rate, -1, 1) * self.max_angular_vel
        yaw_rate = np.clip(yaw_rate, -1, 1) * self.max_angular_vel
        
        # Get current state
        position = self.get_position()
        velocity = self.get_velocity()
        orientation = self.get_orientation()
        angular_velocity = self.get_angular_velocity()
        
        # Calculate aerodynamic forces
        airspeed = np.linalg.norm(velocity)
        if airspeed > 0:
            # Calculate angle of attack and sideslip
            velocity_dir = velocity / airspeed
            roll, pitch, yaw = orientation
            
            # Calculate rotation matrix
            R = self._get_rotation_matrix(roll, pitch, yaw)
            
            # Transform velocity to body frame
            body_velocity = R.T @ velocity
            
            # Calculate aerodynamic forces in body frame
            lift, drag, side_force = self._calculate_aerodynamic_forces(body_velocity)
            
            # Transform forces to global frame
            forces = R @ np.array([-drag, side_force, -lift])
        else:
            forces = np.zeros(3)
        
        # Add thrust force
        forces[2] += thrust
        
        # Add gravity
        forces[2] -= self.mass * 9.81
        
        # Calculate linear acceleration
        acceleration = forces / self.mass
        
        # Update position and velocity
        position += velocity * dt
        velocity += acceleration * dt
        
        # Calculate moments
        moments = self._calculate_moments(roll_rate, pitch_rate, yaw_rate)
        
        # Calculate angular acceleration
        angular_acceleration = moments / self.moment_of_inertia
        
        # Update orientation and angular velocity
        angular_velocity += angular_acceleration * dt
        orientation += angular_velocity * dt
        
        # Update state
        self.state = np.concatenate([position, velocity, orientation, angular_velocity])
    
    def _get_rotation_matrix(self, roll, pitch, yaw):
        """Calculate rotation matrix from Euler angles."""
        # Roll rotation
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # Pitch rotation
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # Yaw rotation
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        return R_z @ R_y @ R_x
    
    def _calculate_aerodynamic_forces(self, body_velocity):
        """Calculate aerodynamic forces in body frame."""
        # Calculate dynamic pressure
        airspeed = np.linalg.norm(body_velocity)
        q = 0.5 * self.air_density * airspeed**2
        
        # Calculate angle of attack and sideslip
        if airspeed > 0:
            alpha = np.arctan2(body_velocity[2], body_velocity[0])  # Angle of attack
            beta = np.arcsin(body_velocity[1] / airspeed)  # Sideslip angle
        else:
            alpha = 0
            beta = 0
        
        # Calculate lift coefficient with stall effects
        if abs(alpha) > self.stall_angle:
            # Simplified stall model
            Cl = self.lift_coefficient * np.sign(alpha) * (1 - (abs(alpha) - self.stall_angle) / np.pi)
        else:
            Cl = self.lift_coefficient * alpha / self.stall_angle
        
        # Calculate forces
        lift = q * self.wing_area * Cl
        drag = q * self.wing_area * self.drag_coefficient
        side_force = q * self.wing_area * self.drag_coefficient * beta
        
        return lift, drag, side_force
    
    def _calculate_moments(self, roll_rate, pitch_rate, yaw_rate):
        """Calculate moments from control inputs and damping."""
        # Control moments
        control_moments = np.array([
            roll_rate * self.moment_of_inertia[0],
            pitch_rate * self.moment_of_inertia[1],
            yaw_rate * self.moment_of_inertia[2]
        ])
        
        # Damping moments (simplified)
        damping = 0.1
        damping_moments = -damping * self.get_angular_velocity()
        
        return control_moments + damping_moments 