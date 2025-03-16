import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os
import time

# Add the parent directory to the path so we can import the drone_swarm_rl package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone_swarm_rl.waypoint_env import WaypointDroneEnv

def main():
    """
    Example of using the return_fig option to create custom visualizations.
    This example adds additional information to the plot and creates a custom layout.
    """
    # Create the waypoint environment with 3 drones and 5 waypoints
    env = WaypointDroneEnv(
        num_drones=3,
        max_steps=1000,
        num_waypoints=5,
        waypoint_size=15.0,
        random_waypoints=False
    )
    
    # Reset the environment
    obs, info = env.reset()
    
    # Create a figure with a custom layout: 3D view and metrics panel
    fig = plt.figure(figsize=(16, 8))
    
    # Create a grid layout
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    
    # Create the 3D axis for the environment
    ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
    
    # Create an axis for metrics
    ax_metrics = fig.add_subplot(gs[0, 1])
    ax_metrics.set_title("Flight Metrics")
    ax_metrics.set_xlim(0, 1000)  # Assuming max 1000 steps
    ax_metrics.set_ylim(-200, 200)
    ax_metrics.set_xlabel("Step")
    ax_metrics.set_ylabel("Value")
    ax_metrics.grid(True)
    
    # Initialize metrics data
    steps = []
    rewards = []
    altitudes = []
    speeds = []
    distances_to_waypoint = []
    
    # Create empty line objects for metrics
    reward_line, = ax_metrics.plot([], [], 'b-', label='Reward')
    altitude_line, = ax_metrics.plot([], [], 'g-', label='Altitude')
    speed_line, = ax_metrics.plot([], [], 'r-', label='Speed')
    distance_line, = ax_metrics.plot([], [], 'y-', label='Distance to Waypoint')
    ax_metrics.legend()
    
    # Text annotations for current values
    step_text = ax_metrics.text(0.05, 0.95, "", transform=ax_metrics.transAxes, verticalalignment='top')
    reward_text = ax_metrics.text(0.05, 0.90, "", transform=ax_metrics.transAxes, verticalalignment='top')
    waypoint_text = ax_metrics.text(0.05, 0.85, "", transform=ax_metrics.transAxes, verticalalignment='top')
    
    # Run the simulation
    total_reward = 0
    step_count = 0
    waypoints_reached = 0
    
    print("\nStarting navigation with custom visualization...")
    
    while True:
        step_count += 1
        
        # Get actions that direct the drones toward the current waypoint
        action = get_waypoint_directed_action(env)
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Get the figure and axes from the environment but don't display yet
        env.fig = fig  # Use our custom figure
        env.ax = ax_3d  # Use our custom 3D axis
        _, _ = env.render(return_fig=True)
        
        # Calculate metrics
        positions = np.array([env.state[f'drone_{i}'][:3] for i in range(env.num_drones)])
        velocities = np.array([env.state[f'drone_{i}'][3:6] for i in range(env.num_drones)])
        centroid = positions.mean(axis=0)
        altitude = centroid[2]
        avg_speed = np.mean([np.linalg.norm(v) for v in velocities])
        
        # Calculate distance to current waypoint
        if env.current_waypoint_idx < len(env.waypoints):
            current_waypoint = env.waypoints[env.current_waypoint_idx]
            distance_to_waypoint = np.linalg.norm(centroid - current_waypoint)
        else:
            distance_to_waypoint = 0
        
        # Update metrics data
        steps.append(step_count)
        rewards.append(reward)
        altitudes.append(altitude)
        speeds.append(avg_speed)
        distances_to_waypoint.append(distance_to_waypoint)
        
        # Update metric lines
        reward_line.set_data(steps, rewards)
        altitude_line.set_data(steps, altitudes)
        speed_line.set_data(steps, speeds)
        distance_line.set_data(steps, distances_to_waypoint)
        
        # Update text annotations
        step_text.set_text(f"Step: {step_count}")
        reward_text.set_text(f"Total Reward: {total_reward:.2f}")
        waypoint_text.set_text(f"Waypoint: {info['current_waypoint'] + 1}/{len(env.waypoints)}")
        
        # Adjust x-axis limit to show all data
        if step_count > 100:
            ax_metrics.set_xlim(step_count - 100, step_count + 10)
        
        # Draw the updated figure
        plt.draw()
        plt.pause(0.01)
        
        # Print progress information when a new waypoint is reached
        if info['waypoints_reached'] > waypoints_reached:
            waypoints_reached = info['waypoints_reached']
            print(f"Step {step_count}: Reached waypoint {waypoints_reached}! Total reward: {total_reward:.2f}")
        
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
    
    # Keep the final plot open until user closes it
    print("\nSimulation complete. Close the plot window to exit.")
    plt.tight_layout()
    plt.show()

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