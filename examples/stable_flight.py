import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path so we can import the drone_swarm_rl package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone_swarm_rl.environment import DroneSwarmEnv

def main():
    # Create the environment
    env = DroneSwarmEnv(num_drones=3, max_steps=1000)
    
    # Reset the environment
    obs, _ = env.reset()
    
    # Run the simulation
    for step in range(500):
        # Get stabilizing actions instead of random actions
        action = env.get_stabilizing_action()
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the environment
        env.render()
        
        if terminated or truncated:
            break
    
    print("Simulation complete!")
    plt.close()

if __name__ == "__main__":
    main() 