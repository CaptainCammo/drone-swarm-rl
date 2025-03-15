import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

# Add the parent directory to the path so we can import the drone_swarm_rl package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone_swarm_rl.environment import DroneSwarmEnv

def main():
    # Create the environment
    env = DroneSwarmEnv(num_drones=3, max_steps=1000)
    
    # Reset the environment
    obs, _ = env.reset()
    
    # Phase 1: Stabilize and fly straight for 50 steps
    print("Phase 1: Flying straight...")
    for step in range(30):  
        # Get stabilizing actions
        action = env.get_stabilizing_action()
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the environment
        env.render()
        
        if terminated or truncated:
            print("Terminated during initial flight!")
            break
    
    # Phase 2: Turn 90 degrees (turn right to face positive y-axis)
    print("Phase 2: Turning 90 degrees...")
    
    # Record the initial yaw angles of all drones
    initial_yaws = []
    for i in range(env.num_drones):
        drone_id = f'drone_{i}'
        initial_yaws.append(env.state[drone_id][8])  # Yaw is at index 8
    
    # Calculate the target yaw angles (90 degrees clockwise from initial)
    target_yaws = []
    for yaw in initial_yaws:
        # Add π/2 and normalize to [-π, π]
        target_yaw = yaw + np.pi/2
        if target_yaw > np.pi:
            target_yaw -= 2 * np.pi
        target_yaws.append(target_yaw)
    
    # Turn until all drones are within the error tolerance of their target yaw
    turn_complete = False
    max_turn_steps = 100  # Safety limit
    turn_steps = 0
    error_tolerance = 0.05  # Radians (about 2.9 degrees)
    
    while not turn_complete and turn_steps < max_turn_steps:
        turn_steps += 1
        
        # Create custom actions for turning while maintaining altitude
        action = {}
        for i in range(env.num_drones):
            drone_id = f'drone_{i}'
            state = env.state[drone_id]
            
            # Extract current state
            orientation = state[6:9]  # [roll, pitch, yaw]
            angular_vel = state[9:12]
            current_yaw = orientation[2]
            
            # Calculate yaw error (how far we still need to turn)
            yaw_diff = np.arctan2(np.sin(target_yaws[i] - current_yaw), np.cos(target_yaws[i] - current_yaw))
            
            # Adaptive yaw rate - turn faster when far from target, slower when close
            # This prevents overshooting
            yaw_rate = np.clip(yaw_diff * 1.0, -0.4, 0.4)
            
            # Keep wings level (roll = 0)
            roll_rate = -orientation[0] * 0.8 - angular_vel[0] * 0.4
            
            # Maintain pitch for altitude
            pitch_rate = (-0.05 - orientation[1]) * 0.8 - angular_vel[1] * 0.4
            
            # Set thrust to maintain speed during turn
            thrust = 0.6
            
            # Combine control inputs
            action[drone_id] = np.array([thrust, roll_rate, pitch_rate, yaw_rate], dtype=np.float32)
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the environment
        env.render()
        
        if terminated or truncated:
            print("Terminated during turn!")
            break
        
        # Check if all drones have completed their turns
        all_drones_turned = True
        current_yaws = []
        
        for i in range(env.num_drones):
            drone_id = f'drone_{i}'
            current_yaw = env.state[drone_id][8]
            current_yaws.append(current_yaw)
            
            # Calculate the angular difference (accounting for wrap-around)
            yaw_diff = np.abs(np.arctan2(np.sin(target_yaws[i] - current_yaw), np.cos(target_yaws[i] - current_yaw)))
            
            if yaw_diff > error_tolerance:
                all_drones_turned = False
        
        # Print progress every 5 steps
        if turn_steps % 5 == 0:
            avg_error = np.mean([np.abs(np.arctan2(np.sin(t - c), np.cos(t - c))) 
                                for t, c in zip(target_yaws, current_yaws)])
            print(f"Turn step {turn_steps}, average error: {avg_error:.4f} radians ({np.degrees(avg_error):.2f} degrees)")
            
            # Also print altitude to monitor
            positions = np.array([env.state[f'drone_{i}'][:3] for i in range(env.num_drones)])
            avg_altitude = np.mean(positions[:, 2])
            print(f"Average altitude: {avg_altitude:.2f}")
        
        turn_complete = all_drones_turned
    
    print(f"Turn completed in {turn_steps} steps")
    
    # Phase 3: Fly straight in the new direction until out of bounds
    print("Phase 3: Flying straight until out of bounds...")
    
    # Create a custom action that maintains the new heading (positive y-axis)
    step_count = 0
    while True:
        step_count += 1
        
        # Create actions for each drone
        action = {}
        for i in range(env.num_drones):
            drone_id = f'drone_{i}'
            state = env.state[drone_id]
            
            # Extract orientation
            orientation = state[6:9]  # [roll, pitch, yaw]
            angular_vel = state[9:12]
            
            # Set thrust to maintain speed
            thrust = 0.7  # Higher thrust to ensure forward movement
            
            # Keep wings level (roll = 0)
            roll_rate = -orientation[0] * 0.8 - angular_vel[0] * 0.4
            
            # Maintain slight negative pitch for lift
            pitch_rate = (-0.05 - orientation[1]) * 0.8 - angular_vel[1] * 0.4
            
            # Maintain yaw at π/2 (facing positive y-axis)
            target_yaw = np.pi/2
            yaw_error = np.arctan2(np.sin(target_yaw - orientation[2]), np.cos(target_yaw - orientation[2]))
            yaw_rate = yaw_error * 0.8 - angular_vel[2] * 0.4
            
            # Combine control inputs
            action[drone_id] = np.array([thrust, roll_rate, pitch_rate, yaw_rate], dtype=np.float32)
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the environment
        env.render()
        
        # Check if we've gone out of bounds
        if terminated or truncated:
            print(f"Terminated after {step_count} steps in the new direction!")
            # Calculate positions and centroid
            positions = np.array([env.state[f'drone_{i}'][:3] for i in range(env.num_drones)])
            centroid = positions.mean(axis=0)
            print(f"Final centroid position: {centroid}")
            break
        
        # Limit the number of steps to prevent infinite loops
        if step_count > 200:
            print("Reached maximum steps without going out of bounds.")
            break
    
    print("Simulation complete!")
    plt.close()

if __name__ == "__main__":
    main() 