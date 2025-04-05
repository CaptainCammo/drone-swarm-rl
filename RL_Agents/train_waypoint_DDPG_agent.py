import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import os
from datetime import datetime
from drone_swarm_rl.agents.ddpg_agent_pytorch import DroneSwarmDDPG
from drone_swarm_rl.waypoint_env import WaypointDroneEnv, create_scanning_waypoints
import argparse

def train_waypoint_agent(episodes=100, mode='train', render=False, render_delay=0.01, render_interval=10):
    """
    Train a DDPG agent to navigate through waypoints in a scanning pattern.
    
    Args:
        episodes: Number of training episodes
        mode: 'train' or 'test'
        render: Whether to render the environment during training
        render_delay: Delay between renders
        render_interval: How often to render (every N episodes)
    """
    # Initialize timestamp at the beginning of training
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_type = "DDPG"  # Add agent type identifier
    
    # Set matplotlib backend for rendering
    if render:
        plt.switch_backend('Agg')
        plt.ion()
    
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
    
    # Create agent with DDPG-specific hyperparameters
    agent = DroneSwarmDDPG(
        observation_space=env.observation_space,
        action_space=env.action_space,
        actor_lr=0.001,  # Learning rate for actor network
        critic_lr=0.001,  # Learning rate for critic network
        gamma=0.99,      # Discount factor
        tau=0.001,       # Soft target update parameter
        batch_size=64,   # Batch size for experience replay
        memory_size=100000  # Size of replay buffer
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
            # Select action with noise for exploration
            action = agent.act(combined_state, add_noise=True)
            
            # Convert action tensor to dictionary format
            action_dict = {}
            for i, drone_id in enumerate(sorted(state.keys())):
                # Get the action for this drone (assuming action is a 3D tensor [batch, num_drones, num_actions])
                drone_action = action[0, i]  # Remove batch dimension and select drone
                
                # Convert to numpy array if it's a tensor
                if torch.is_tensor(drone_action):
                    drone_action = drone_action.cpu().detach().numpy()
                
                # Ensure we have scalar values for each action component
                action_dict[drone_id] = np.array([
                    float(drone_action[0]),  # thrust
                    float(drone_action[1]),  # roll_rate
                    float(drone_action[2]),  # pitch_rate
                    float(drone_action[3])   # yaw_rate
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
                # Clear previous figure if it exists
                if hasattr(env, 'fig') and env.fig is not None:
                    plt.close(env.fig)
                env.render()
                plt.pause(render_delay)
        
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
        if mode == 'train' and (episode + 1) % 200 == 0:
            # Create directory for this training run
            save_dir = os.path.join('models', f'{agent_type}_training_{timestamp}')
            os.makedirs(save_dir, exist_ok=True)
            
            # Save model with episode number
            model_path = os.path.join(save_dir, f'{agent_type.lower()}_agent_episode_{episode+1}.pt')
            agent.save(model_path)
            print(f"Model saved to {model_path}")
            
            # Save metrics and plots for this checkpoint
            save_metrics(metrics, timestamp, agent_type)
            plot_training_metrics(metrics, timestamp, agent_type)
    
    # Save final model
    if mode == 'train':
        # Create directory for this training run
        save_dir = os.path.join('models', f'{agent_type}_training_{timestamp}')
        os.makedirs(save_dir, exist_ok=True)
        
        model_path = os.path.join(save_dir, f'{agent_type.lower()}_agent_final.pt')
        agent.save(model_path)
        print(f"Final model saved to {model_path}")
        
        # Save final metrics and plots
        save_metrics(metrics, timestamp, agent_type)
        plot_training_metrics(metrics, timestamp, agent_type)
    
    # Save metrics
    metrics['termination_reasons'] = termination_reasons
    save_metrics(metrics, timestamp, agent_type)
    
    # Plot metrics
    plot_training_metrics(metrics, timestamp, agent_type)
    
    # Close any remaining figures
    if render:
        plt.close('all')
        plt.ioff()  # Turn off interactive mode
    
    return agent

def test_trained_agent(model_path, num_episodes=5, render=True, render_delay=0.01):
    """Test a trained DDPG agent on the waypoint navigation task."""
    print(f"\nTesting trained agent using model: {model_path}")
    
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
    agent = DroneSwarmDDPG(
        observation_space=env.observation_space,
        action_space=env.action_space,
        actor_lr=0.001,
        critic_lr=0.001,
        gamma=0.99,
        tau=0.001,
        batch_size=64,
        memory_size=100000
    )
    
    # Load trained model
    agent.load(model_path)
    
    # Set matplotlib backend for rendering
    if render:
        plt.switch_backend('Agg')
        plt.ion()
    
    # Test loop
    for episode in range(num_episodes):
        print(f"\nStarting test episode {episode+1}/{num_episodes}")
        state, _ = env.reset()
        
        # Convert state to tensor format
        state_tensor = {}
        for drone_id, obs in state.items():
            state_tensor[drone_id] = torch.FloatTensor(obs)
        
        # Concatenate all drone states
        combined_state = torch.cat([state_tensor[drone_id] for drone_id in sorted(state_tensor.keys())])
        
        episode_reward = 0
        done = False
        
        while not done:
            # Select action without noise for testing
            action = agent.act(combined_state, add_noise=False)
            
            # Convert action tensor to dictionary format
            action_dict = {}
            for i, drone_id in enumerate(sorted(state.keys())):
                # Get the action for this drone (assuming action is a 3D tensor [batch, num_drones, num_actions])
                drone_action = action[0, i]  # Remove batch dimension and select drone
                
                # Convert to numpy array if it's a tensor
                if torch.is_tensor(drone_action):
                    drone_action = drone_action.cpu().detach().numpy()
                
                # Ensure we have scalar values for each action component
                action_dict[drone_id] = np.array([
                    float(drone_action[0]),  # thrust
                    float(drone_action[1]),  # roll_rate
                    float(drone_action[2]),  # pitch_rate
                    float(drone_action[3])   # yaw_rate
                ], dtype=np.float32)
            
            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action_dict)
            done = terminated or truncated
            
            # Convert next state to tensor format
            next_state_tensor = {}
            for drone_id, obs in next_state.items():
                next_state_tensor[drone_id] = torch.FloatTensor(obs)
            
            # Concatenate all next drone states
            combined_state = torch.cat([next_state_tensor[drone_id] for drone_id in sorted(next_state_tensor.keys())])
            
            episode_reward += reward
            
            # Render if requested
            if render:
                # Clear previous figure if it exists
                if hasattr(env, 'fig') and env.fig is not None:
                    plt.close(env.fig)
                env.render()
                plt.pause(render_delay)
        
        # Print episode results
        print(f"Episode {episode+1} completed")
        print(f"Total reward: {episode_reward:.2f}")
        print(f"Final waypoint index: {env.current_waypoint_index}/{len(waypoints)}")
        if 'terminated_reason' in info:
            print(f"Termination reason: {info['terminated_reason']}")
    
    # Close any remaining figures
    if render:
        plt.close('all')
        plt.ioff()

def save_metrics(metrics, timestamp=None, agent_type="DDPG"):
    """Save metrics to a file for later analysis."""
    # Use provided timestamp or create a new one
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory for metrics
    metrics_dir = os.path.join('training_results', f'{agent_type}_training_{timestamp}')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Save metrics to file
    metrics_path = os.path.join(metrics_dir, 'metrics.npz')
    np.savez(metrics_path, **metrics)
    print(f"Metrics saved to {metrics_path}")

def plot_training_metrics(metrics, timestamp=None, agent_type="DDPG"):
    """Plot and save training metrics."""
    # Use provided timestamp or create a new one
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory for plots
    plot_dir = os.path.join('training_results', f'{agent_type}_training_{timestamp}')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot episode rewards
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['episode_rewards'])
    plt.title(f'{agent_type} Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(os.path.join(plot_dir, 'episode_rewards.png'))
    plt.close()
    
    # Plot episode lengths
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['episode_lengths'])
    plt.title(f'{agent_type} Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.savefig(os.path.join(plot_dir, 'episode_lengths.png'))
    plt.close()
    
    # Plot waypoints reached
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['waypoints_reached'])
    plt.title(f'{agent_type} Waypoints Reached')
    plt.xlabel('Episode')
    plt.ylabel('Waypoints')
    plt.savefig(os.path.join(plot_dir, 'waypoints_reached.png'))
    plt.close()
    
    # Plot reward components if available
    if metrics['formation_quality_rewards']:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['formation_quality_rewards'])
        plt.title(f'{agent_type} Formation Quality Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig(os.path.join(plot_dir, 'formation_quality_rewards.png'))
        plt.close()
    
    if metrics['velocity_alignment_rewards']:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['velocity_alignment_rewards'])
        plt.title(f'{agent_type} Velocity Alignment Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig(os.path.join(plot_dir, 'velocity_alignment_rewards.png'))
        plt.close()
    
    if metrics['waypoint_rewards']:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['waypoint_rewards'])
        plt.title(f'{agent_type} Waypoint Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig(os.path.join(plot_dir, 'waypoint_rewards.png'))
        plt.close()
    
    if metrics['distance_rewards']:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['distance_rewards'])
        plt.title(f'{agent_type} Distance Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig(os.path.join(plot_dir, 'distance_rewards.png'))
        plt.close()
    
    if metrics['wrong_direction_penalties']:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['wrong_direction_penalties'])
        plt.title(f'{agent_type} Wrong Direction Penalties')
        plt.xlabel('Episode')
        plt.ylabel('Penalty')
        plt.savefig(os.path.join(plot_dir, 'wrong_direction_penalties.png'))
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
        plt.title(f'{agent_type} Termination Reasons')
        plt.xlabel('Reason')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'termination_reasons.png'))
        plt.close()
    
    # Plot combined metrics
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(metrics['episode_rewards'])
    plt.title(f'{agent_type} Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(2, 2, 2)
    plt.plot(metrics['episode_lengths'])
    plt.title(f'{agent_type} Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.subplot(2, 2, 3)
    plt.plot(metrics['waypoints_reached'])
    plt.title(f'{agent_type} Waypoints Reached')
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
        plt.title(f'{agent_type} Termination Reasons')
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'combined_metrics.png'))
    plt.close()
    
    print(f"Training plots saved to {plot_dir}/")

def main():
    """Main function to train and test the DDPG agent."""
    # Create argument parser
    parser = argparse.ArgumentParser(description='Train and test a DDPG agent for drone swarm waypoint navigation.')
    
    # Training arguments
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes (default: 1000)')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help='Mode: train or test (default: train)')
    parser.add_argument('--render', action='store_true',
                        help='Whether to render the environment during training')
    parser.add_argument('--render_delay', type=float, default=0.01,
                        help='Delay between renders in seconds (default: 0.01)')
    parser.add_argument('--render_interval', type=int, default=10,
                        help='How often to render (every N episodes) (default: 10)')
    
    # Testing arguments
    parser.add_argument('--test_episodes', type=int, default=5,
                        help='Number of test episodes (default: 5)')
    parser.add_argument('--model_path', type=str, default='models/DDPG_training_*/ddpg_agent_final.pt',
                        help='Path to the model to test (default: models/DDPG_training_*/ddpg_agent_final.pt)')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Train the agent
        agent = train_waypoint_agent(
            episodes=args.episodes,
            mode=args.mode,
            render=args.render,
            render_delay=args.render_delay,
            render_interval=args.render_interval
        )
        
        # Test the trained agent
        test_trained_agent(
            model_path=args.model_path,
            num_episodes=args.test_episodes,
            render=True,
            render_delay=args.render_delay
        )
    else:
        # Only test the agent
        test_trained_agent(
            model_path=args.model_path,
            num_episodes=args.test_episodes,
            render=True,
            render_delay=args.render_delay
        )

if __name__ == "__main__":
    main() 