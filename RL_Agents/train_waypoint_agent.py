import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
import time
from datetime import datetime
import json

# Add the parent directory to the path so we can import the drone_swarm_rl package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone_swarm_rl.waypoint_env import WaypointDroneEnv
from drone_swarm_rl.agents.dqn_agent_pytorch import DroneSwarmDQN

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

def train_waypoint_agent(episodes=100, mode='train', render=False, render_delay=0.01, render_interval=10):
    """
    Train a DQN agent to navigate through 10 ordered waypoints in a scanning pattern.
    
    Args:
        episodes: Number of training episodes
        mode: 'train' or 'test'
        render: Whether to render the environment during training
        render_delay: Delay between renders
        render_interval: How often to render (every N episodes)
    """
    # Create environment
    waypoints = create_scanning_waypoints()
    env = WaypointDroneEnv(
        num_drones=3,
        max_steps=5000,
        waypoint_path=waypoints,
        waypoint_size=60.0,
        waypoint_reward=150.0,
        distance_reward_factor=0.3,
        max_steps_away=50,
        wrong_direction_penalty=200.0,
        random_waypoints=False,
        num_waypoints=2
    )
    
    # Create agent
    agent = DroneSwarmDQN(
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.998,
        batch_size=64,
        target_update=10,
        memory_size=100000
    )
    
    # Initialize metrics
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'waypoints_reached': [],
        'formation_quality_rewards': [],
        'velocity_alignment_rewards': [],
        'waypoint_rewards': [],
        'distance_rewards': [],
        'wrong_direction_penalties': []
    }
    
    # Initialize termination reasons tracking
    termination_reasons = {
        'all_waypoints_reached': 0,
        'moving_away_too_long': 0,
        'out_of_bounds_x_min': 0,
        'out_of_bounds_x_max': 0,
        'out_of_bounds_y_min': 0,
        'out_of_bounds_y_max': 0,
        'out_of_bounds_z_min': 0,
        'out_of_bounds_z_max': 0,
        'max_steps': 0,
        'other': 0
    }
    
    # Training loop
    for episode in range(episodes):
        # Reset environment
        state, _ = env.reset()
        
        # Convert state to tensor format
        state_tensor = {}
        for drone_id, obs in state.items():
            state_tensor[drone_id] = torch.FloatTensor(obs)
        
        # Concatenate all drone states
        combined_state = torch.cat([state_tensor[drone_id] for drone_id in sorted(state_tensor.keys())])
        
        # Initialize episode metrics
        episode_reward = 0
        episode_length = 0
        episode_formation_quality_rewards = []
        episode_velocity_alignment_rewards = []
        episode_waypoint_rewards = []
        episode_distance_rewards = []
        episode_wrong_direction_penalties = []
        
        # Episode loop
        done = False
        while not done:
            # Select action
            action = agent.act(combined_state)
            
            # Convert action tensor to dictionary format
            action_dict = {}
            for i, drone_id in enumerate(sorted(state.keys())):
                # Convert tensor to numpy array and ensure it has the correct number of values
                drone_action = action[i].cpu().numpy()
                expected_action_size = env.action_space[drone_id].shape[0]
                if len(drone_action) != expected_action_size:
                    # If action is missing dimensions, pad with zeros
                    drone_action = np.pad(drone_action, (0, expected_action_size - len(drone_action)))
                # Ensure we have scalar values for each action component
                # Handle batch dimension by taking the first element
                action_dict[drone_id] = np.array([
                    float(drone_action[0][0]),  # thrust
                    float(drone_action[0][1]),  # roll_rate
                    float(drone_action[0][2]),  # pitch_rate
                    float(drone_action[0][3])   # yaw_rate
                ], dtype=np.float32)
            
            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action_dict)
            done = terminated or truncated
            
            # Convert next state to tensor format
            next_state_tensor = {}
            for drone_id, obs in next_state.items():
                next_state_tensor[drone_id] = torch.FloatTensor(obs)
            
            # Concatenate all next drone states
            combined_next_state = torch.cat([next_state_tensor[drone_id] for drone_id in sorted(next_state_tensor.keys())])
            
            # Store transition
            agent.remember(combined_state, action, reward, combined_next_state, done)
            agent.replay()
            
            # Update state
            combined_state = combined_next_state
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            
            # Track reward components
            if 'formation_quality_reward' in info:
                episode_formation_quality_rewards.append(info['formation_quality_reward'])
            if 'velocity_alignment_reward' in info:
                episode_velocity_alignment_rewards.append(info['velocity_alignment_reward'])
            if 'waypoint_reward' in info:
                episode_waypoint_rewards.append(info['waypoint_reward'])
            if 'distance_reward' in info:
                episode_distance_rewards.append(info['distance_reward'])
            if 'wrong_direction_penalty' in info:
                episode_wrong_direction_penalties.append(info['wrong_direction_penalty'])
            
            # Track termination reason
            if 'terminated_reason' in info:
                reason = info['terminated_reason']
                if reason in termination_reasons:
                    termination_reasons[reason] += 1
                else:
                    termination_reasons['other'] += 1
            
            # Render if requested and at the specified interval
            if render and episode % render_interval == 0:
                env.render()
                time.sleep(render_delay)
        
        # Update metrics
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_lengths'].append(episode_length)
        metrics['waypoints_reached'].append(env.current_waypoint_index)
        
        # Update reward component metrics
        if episode_formation_quality_rewards:
            metrics['formation_quality_rewards'].append(sum(episode_formation_quality_rewards))
        if episode_velocity_alignment_rewards:
            metrics['velocity_alignment_rewards'].append(sum(episode_velocity_alignment_rewards))
        if episode_waypoint_rewards:
            metrics['waypoint_rewards'].append(sum(episode_waypoint_rewards))
        if episode_distance_rewards:
            metrics['distance_rewards'].append(sum(episode_distance_rewards))
        if episode_wrong_direction_penalties:
            metrics['wrong_direction_penalties'].append(sum(episode_wrong_direction_penalties))
        
        # Print episode stats
        print(f"Episode {episode+1}/{episodes}, Reward: {episode_reward:.2f}, Length: {episode_length}, Waypoints: {env.current_waypoint_index}/{len(waypoints)}")
        
        # Print termination reason
        if 'terminated_reason' in info:
            print(f"  Termination reason: {info['terminated_reason']}")
        
        # Save model periodically
        if mode == 'train' and (episode + 1) % 100 == 0:
            model_path = os.path.join('models', f'dqn_agent_episode_{episode+1}.pt')
            agent.save(model_path)
            print(f"Model saved to {model_path}")
    
    # Save final model
    if mode == 'train':
        model_path = os.path.join('models', 'dqn_agent_final.pt')
        agent.save(model_path)
        print(f"Final model saved to {model_path}")
    
    # Save metrics
    metrics['termination_reasons'] = termination_reasons
    save_metrics(metrics)
    
    # Plot metrics
    plot_training_metrics(metrics)
    
    return agent

def plot_training_metrics(metrics):
    """Plot and save training metrics."""
    # Create directory for plots if it doesn't exist
    os.makedirs('training_results', exist_ok=True)
    
    # Plot episode rewards
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['episode_rewards'])
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig('training_results/episode_rewards.png')
    plt.close()
    
    # Plot episode lengths
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['episode_lengths'])
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.savefig('training_results/episode_lengths.png')
    plt.close()
    
    # Plot waypoints reached
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['waypoints_reached'])
    plt.title('Waypoints Reached')
    plt.xlabel('Episode')
    plt.ylabel('Waypoints')
    plt.savefig('training_results/waypoints_reached.png')
    plt.close()
    
    # Plot reward components if available
    if metrics['formation_quality_rewards']:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['formation_quality_rewards'])
        plt.title('Formation Quality Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig('training_results/formation_quality_rewards.png')
        plt.close()
    
    if metrics['velocity_alignment_rewards']:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['velocity_alignment_rewards'])
        plt.title('Velocity Alignment Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig('training_results/velocity_alignment_rewards.png')
        plt.close()
    
    if metrics['waypoint_rewards']:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['waypoint_rewards'])
        plt.title('Waypoint Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig('training_results/waypoint_rewards.png')
        plt.close()
    
    if metrics['distance_rewards']:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['distance_rewards'])
        plt.title('Distance Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig('training_results/distance_rewards.png')
        plt.close()
    
    if metrics['wrong_direction_penalties']:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['wrong_direction_penalties'])
        plt.title('Wrong Direction Penalties')
        plt.xlabel('Episode')
        plt.ylabel('Penalty')
        plt.savefig('training_results/wrong_direction_penalties.png')
        plt.close()
    
    # Plot termination reasons
    if 'termination_reasons' in metrics:
        plt.figure(figsize=(12, 6))
        reasons = list(metrics['termination_reasons'].keys())
        counts = list(metrics['termination_reasons'].values())
        
        # Sort by count (descending)
        sorted_indices = np.argsort(counts)[::-1]
        sorted_reasons = [reasons[i] for i in sorted_indices]
        sorted_counts = [counts[i] for i in sorted_indices]
        
        plt.bar(sorted_reasons, sorted_counts)
        plt.title('Termination Reasons')
        plt.xlabel('Reason')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('training_results/termination_reasons.png')
        plt.close()
    
    # Plot combined metrics
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(metrics['episode_rewards'])
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(2, 2, 2)
    plt.plot(metrics['episode_lengths'])
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.subplot(2, 2, 3)
    plt.plot(metrics['waypoints_reached'])
    plt.title('Waypoints Reached')
    plt.xlabel('Episode')
    plt.ylabel('Waypoints')
    
    plt.subplot(2, 2, 4)
    if 'termination_reasons' in metrics:
        reasons = list(metrics['termination_reasons'].keys())
        counts = list(metrics['termination_reasons'].values())
        
        # Sort by count (descending)
        sorted_indices = np.argsort(counts)[::-1]
        sorted_reasons = [reasons[i] for i in sorted_indices]
        sorted_counts = [counts[i] for i in sorted_indices]
        
        plt.bar(sorted_reasons, sorted_counts)
        plt.title('Termination Reasons')
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('training_results/combined_metrics.png')
    plt.close()

def test_trained_agent(model_path, num_episodes=5, render=False):
    """Test a trained agent on the waypoint navigation task."""
    # Create scanning waypoints
    waypoints = create_scanning_waypoints()
    
    # Create environment
    env = WaypointDroneEnv(
        num_drones=3,
        max_steps=2000,
        num_waypoints=10,
        waypoint_size=15.0,
        random_waypoints=False,
        waypoint_path=waypoints
    )
    
    # Create agent and load trained weights
    agent = DroneSwarmDQN(
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=0.01,  # Set to minimum for testing
        epsilon_end=0.01,
        epsilon_decay=1.0,
        batch_size=64,
        target_update=10,
        memory_size=100000
    )
    agent.load(model_path)
    
    # Test loop
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        waypoints_reached = 0
        
        print(f"\nStarting test episode {episode + 1}/{num_episodes}")
        
        # Convert state to tensor format
        state_tensor = {}
        for drone_id, obs in state.items():
            state_tensor[drone_id] = torch.FloatTensor(obs)
        
        # Concatenate all drone states
        combined_state = torch.cat([state_tensor[drone_id] for drone_id in sorted(state_tensor.keys())])
        
        for step in range(env.max_steps):
            # Get action from agent
            action = agent.act(combined_state, use_epsilon=False)
            
            # Convert action to dictionary format
            action_dict = {}
            for i, drone_id in enumerate(sorted(state.keys())):
                action_dict[drone_id] = action[i].numpy()
            
            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action_dict)
            done = terminated or truncated
            
            total_reward += reward
            waypoints_reached = info['waypoints_reached']
            
            # Convert next state to tensor format
            next_state_tensor = {}
            for drone_id, obs in next_state.items():
                next_state_tensor[drone_id] = torch.FloatTensor(obs)
            
            # Concatenate all next drone states
            combined_state = torch.cat([next_state_tensor[drone_id] for drone_id in sorted(next_state_tensor.keys())])
            
            # Render environment
            if render:
                env.render()
            
            # Add a small delay to make visualization visible
            if step % 5 == 0:
                time.sleep(0.05)
            
            # Print progress
            if step % 100 == 0:
                print(f"Step {step}: Waypoints reached: {waypoints_reached}/{len(waypoints)}")
            
            # Break if episode is done
            if done:
                plt.close()
                break
        
        print(f"Test episode {episode + 1} complete")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Waypoints Reached: {waypoints_reached}/{len(waypoints)}")
        print(f"Steps: {step + 1}")

def save_metrics(metrics):
    """Save metrics to a file for later analysis."""
    # Create directory for metrics if it doesn't exist
    os.makedirs('training_results', exist_ok=True)
    
    # Save metrics as JSON
    with open('training_results/metrics.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                serializable_metrics[key] = [arr.tolist() for arr in value]
            elif isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            else:
                serializable_metrics[key] = value
        
        json.dump(serializable_metrics, f, indent=4)
    
    # Save individual metrics as numpy arrays for numerical analysis
    for key, value in metrics.items():
        if key != 'termination_reasons' and value:  # Skip dictionary items
            try:
                np.save(f'training_results/{key}.npy', np.array(value))
            except:
                print(f"Could not save {key} as numpy array")
    
    # Save termination reasons as text file for easy reading
    if 'termination_reasons' in metrics:
        with open('training_results/termination_reasons.txt', 'w') as f:
            for reason, count in metrics['termination_reasons'].items():
                f.write(f"{reason}: {count}\n")
    
    print(f"Metrics saved to training_results/")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if we should train or test
    import argparse
    parser = argparse.ArgumentParser(description='Train or test a waypoint navigation agent')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], 
                        help='Whether to train a new agent or test an existing one')
    parser.add_argument('--model', type=str, default=None, 
                        help='Path to model file for testing')
    parser.add_argument('--episodes', type=int, default=300, 
                        help='Number of episodes for training')
    parser.add_argument('--render', type=bool, default=True,
                        help='Whether to render the environment during training')
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Training a new waypoint navigation agent...")
        train_waypoint_agent(episodes=args.episodes, render=args.render)
    else:
        if args.model is None:
            print("Error: Must provide a model path for testing")
            sys.exit(1)
        print(f"Testing trained agent using model: {args.model}")
        test_trained_agent(args.model) 