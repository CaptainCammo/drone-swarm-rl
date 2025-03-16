import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

# Add the parent directory to the path so we can import the drone_swarm_rl package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone_swarm_rl.waypoint_env import WaypointDroneEnv

def main():
    # Create the waypoint environment with 3 drones and 5 waypoints
    env = WaypointDroneEnv(
        num_drones=3,
        max_steps=1000,
        num_waypoints=5,
        waypoint_size=15.0,  # Larger boxes for easier navigation
        random_waypoints=False  # Use the default predefined path
    )
    
    # Reset the environment
    obs, info = env.reset()
    
    # Print waypoint information
    print(f"Generated {len(env.waypoints)} waypoints:")
    for i, wp in enumerate(env.waypoints):
        print(f"  Waypoint {i+1}: {wp}")
    
    # Run the simulation
    total_reward = 0
    step_count = 0
    waypoints_reached = 0
    
    print("\nStarting navigation...")
    
    # Example of getting the figure and axes (optional)
    # fig, ax = env.render(return_fig=True)
    # You could customize the figure further here if needed
    # plt.draw()
    # plt.pause(0.01)
    
    while True:
        step_count += 1
        
        # Get stabilizing actions with a bias toward the current waypoint
        action = get_waypoint_directed_action(env)
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the environment (standard way - displays automatically)
        env.render()
        
        # Alternative: get figure and axes for custom modifications
        # fig, ax = env.render(return_fig=True)
        # Add custom elements to the plot if needed
        # plt.draw()
        # plt.pause(0.01)
        
        # Print progress information when a new waypoint is reached
        if info['waypoints_reached'] > waypoints_reached:
            waypoints_reached = info['waypoints_reached']
            print(f"Step {step_count}: Reached waypoint {waypoints_reached}! Total reward: {total_reward:.2f}")
        
        # Print periodic updates
        if step_count % 50 == 0:
            print(f"Step {step_count}: Current waypoint: {info['current_waypoint'] + 1}, "
                  f"Waypoints reached: {info['waypoints_reached']}/{len(env.waypoints)}, "
                  f"Total reward: {total_reward:.2f}")
        
        # Check if episode is done
        if terminated or truncated:
            if info.get('all_waypoints_reached', False):
                print(f"\nSuccess! All waypoints reached in {step_count} steps!")
            else:
                print(f"\nEpisode ended after {step_count} steps. "
                      f"Reached {info['waypoints_reached']}/{len(env.waypoints)} waypoints.")
            break
        
        # Safety limit
        if step_count >= 1000:
            print("\nReached maximum steps.")
            break
    
    print(f"Final total reward: {total_reward:.2f}")
    plt.close()

def get_waypoint_directed_action(env):
    """
    Generate actions that direct the drones toward the current waypoint.
    This is a simple heuristic controller that aims to navigate through waypoints.
    """
    # Get current waypoint
    if env.current_waypoint_idx >= len(env.waypoints):
        # If all waypoints are reached, just stabilize
        return env.get_stabilizing_action()
    
    current_waypoint = env.waypoints[env.current_waypoint_idx]
    
    # Calculate centroid position
    positions = np.array([env.state[f'drone_{i}'][:3] for i in range(env.num_drones)])
    centroid = positions.mean(axis=0)
    
    # Vector from centroid to waypoint
    to_waypoint = current_waypoint - centroid
    distance = np.linalg.norm(to_waypoint)
    
    # If very close to waypoint, just stabilize
    if distance < 5.0:
        return env.get_stabilizing_action()
    
    # Normalize direction vector
    direction = to_waypoint / distance
    
    # Calculate desired yaw (heading) to face the waypoint
    # Note: yaw of 0 is facing positive x-axis, and increases counterclockwise
    desired_yaw = np.arctan2(direction[1], direction[0])
    
    # Create actions for each drone
    action = {}
    for i in range(env.num_drones):
        drone_id = f'drone_{i}'
        state = env.state[drone_id]
        
        # Extract current state
        orientation = state[6:9]  # [roll, pitch, yaw]
        angular_vel = state[9:12]
        current_yaw = orientation[2]
        
        # Calculate yaw error (how much we need to turn)
        yaw_error = np.arctan2(np.sin(desired_yaw - current_yaw), np.cos(desired_yaw - current_yaw))
        
        # Set thrust based on distance and alignment
        thrust = 0.6  # Base thrust
        
        # Increase thrust if well-aligned with target and far away
        alignment = np.cos(yaw_error)
        if alignment > 0.8 and distance > 20:
            thrust = 0.8
        
        # Yaw control to turn toward waypoint
        yaw_rate = yaw_error * 0.8 - angular_vel[2] * 0.4
        
        # Roll control to keep wings level
        roll_rate = -orientation[0] * 0.8 - angular_vel[0] * 0.4
        
        # Pitch control for altitude
        # Adjust pitch based on waypoint height relative to current height
        height_diff = current_waypoint[2] - centroid[2]
        target_pitch = -0.05  # Default slight negative pitch
        
        if height_diff > 5:
            # Need to climb
            target_pitch = -0.15
        elif height_diff < -5:
            # Need to descend
            target_pitch = 0.05
            
        pitch_rate = (target_pitch - orientation[1]) * 0.8 - angular_vel[1] * 0.4
        
        # Combine control inputs
        action[drone_id] = np.array([thrust, roll_rate, pitch_rate, yaw_rate], dtype=np.float32)
    
    return action

if __name__ == "__main__":
    main() 