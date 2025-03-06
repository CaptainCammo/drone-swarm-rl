import numpy as np
from drone_swarm_rl.environment import DroneSwarmEnv

def main():
    # Create the environment with 3 drones
    env = DroneSwarmEnv(num_drones=3)
    
    # Reset the environment
    obs, _ = env.reset()
    
    # Run for 100 steps
    for step in range(100):
        # Create random actions for each drone
        action = {
            drone_id: env.action_space[drone_id].sample()
            for drone_id in env.action_space.spaces
        }
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step}")
        print(f"Reward: {reward}")
        
        # Print position of each drone
        for drone_id, drone_obs in obs.items():
            position = drone_obs[:3]
            print(f"{drone_id} position: {position}")
        
        print("---")
        
        if terminated or truncated:
            break
    
if __name__ == "__main__":
    main() 