import numpy as np
import torch
from drone_swarm_rl.environment import DroneSwarmEnv
from drone_swarm_rl.agents.dqn_agent_pytorch import DroneSwarmDQN

def train_dqn(episodes=1000, max_steps=1000):
    # Create environment and agent
    env = DroneSwarmEnv(num_drones=3, max_steps=max_steps)
    agent = DroneSwarmDQN(state_size=12, action_size=4, num_drones=3)
    
    # Training metrics
    episode_rewards = []
    
    # Training loop
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Get action from agent
            action = agent.act(state)
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience and train
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            total_reward += reward
            state = next_state
            
            # Render environment (comment out for faster training)
            env.render()
            
            if done:
                break
        
        # Store episode reward
        episode_rewards.append(total_reward)
        
        # Calculate running average
        running_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        
        # Print episode statistics
        print(f"Episode: {episode + 1}/{episodes}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Running Average (100 episodes): {running_avg:.2f}")
        print(f"Epsilon: {agent.epsilon:.4f}")
        print("---")
        
        # Save model weights periodically
        if (episode + 1) % 100 == 0:
            agent.save(f"dqn_weights_episode_{episode + 1}.pt")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    train_dqn() 