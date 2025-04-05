import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import copy

class ActorNetwork(nn.Module):
    def __init__(self, num_drones, state_size_per_drone, action_size_per_drone):
        super(ActorNetwork, self).__init__()
        self.num_drones = num_drones
        self.state_size_per_drone = state_size_per_drone
        self.action_size_per_drone = action_size_per_drone
        
        # Calculate total input size
        total_state_size = num_drones * state_size_per_drone
        
        # Define the actor network architecture
        self.layer1 = nn.Sequential(
            nn.Linear(total_state_size, 512),
            nn.ReLU(),
            nn.LayerNorm(512)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.LayerNorm(512)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )
        
        self.layer4 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )
        
        self.layer5 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )
        
        # Output layer - produces actions for all drones
        self.output_layer = nn.Linear(128, num_drones * action_size_per_drone)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        # Ensure input has batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Forward pass through the network
        x = self.layer1(state)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.output_layer(x)
        
        # Apply tanh to bound actions between -1 and 1
        x = torch.tanh(x)
        
        # Reshape output to [batch_size, num_drones, action_size_per_drone]
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_drones, self.action_size_per_drone)
        
        return x

class CriticNetwork(nn.Module):
    def __init__(self, num_drones, state_size_per_drone, action_size_per_drone):
        super(CriticNetwork, self).__init__()
        self.num_drones = num_drones
        self.state_size_per_drone = state_size_per_drone
        self.action_size_per_drone = action_size_per_drone
        
        # Calculate total input sizes
        total_state_size = num_drones * state_size_per_drone
        total_action_size = num_drones * action_size_per_drone
        
        # State processing layers
        self.state_layer1 = nn.Sequential(
            nn.Linear(total_state_size, 512),
            nn.ReLU(),
            nn.LayerNorm(512)
        )
        
        # Action processing layers
        self.action_layer1 = nn.Sequential(
            nn.Linear(total_action_size, 512),
            nn.ReLU(),
            nn.LayerNorm(512)
        )
        
        # Combined processing layers
        self.combined_layer1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LayerNorm(512)
        )
        
        self.combined_layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )
        
        self.combined_layer3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )
        
        # Output layer - produces Q-value
        self.output_layer = nn.Linear(128, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state, action):
        # Ensure inputs have batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        
        # Process state and action
        state_features = self.state_layer1(state)
        action_features = self.action_layer1(action)
        
        # Combine features
        combined = torch.cat([state_features, action_features], dim=1)
        
        # Process combined features
        x = self.combined_layer1(combined)
        x = self.combined_layer2(x)
        x = self.combined_layer3(x)
        x = self.output_layer(x)
        
        return x



# Define Ornstein-Uhlenbeck noise process to allow for early exploration.
class OrnsteinUhlenbeckNoise:
    def __init__(self, size, mu=0, theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.size) * self.mu
    
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state

class DroneSwarmDDPG:
    def __init__(self, observation_space, action_space, 
                 actor_lr=0.001, critic_lr=0.001, gamma=0.99,
                 tau=0.001, batch_size=64, memory_size=100000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get dimensions from spaces
        # Get dimensions from the first drone (assuming all drones have same state/action sizes)
        self.first_drone_id = next(iter(observation_space.spaces.keys()))
        self.state_size_per_drone = observation_space.spaces[self.first_drone_id].shape[0]
        self.action_size_per_drone = action_space.spaces[self.first_drone_id].shape[0]
        self.num_drones = len(observation_space.spaces)
        
        # Create networks
        self.actor = ActorNetwork(self.num_drones, self.state_size_per_drone, self.action_size_per_drone).to(self.device)
        self.critic = CriticNetwork(self.num_drones, self.state_size_per_drone, self.action_size_per_drone).to(self.device)
        
        # Create target networks
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        
        # Create optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Set hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        # Create replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Create noise process
        self.noise = OrnsteinUhlenbeckNoise(
            size=self.num_drones * self.action_size_per_drone,
            mu=0,
            theta=0.15,
            sigma=0.2
        )
    
    def act(self, state, add_noise=True):
        """Select an action using the actor network."""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action = self.actor(state)
            
            if add_noise:
                noise = torch.FloatTensor(self.noise.sample()).to(self.device)
                noise = noise.view_as(action)
                action = action + noise
            
            # Clip actions to valid range
            action = torch.clamp(action, -1, 1)
            
            return action.cpu().numpy()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train the networks using experience replay."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        
        # Convert states and actions to tensors
        states = torch.stack([torch.FloatTensor(x[0]) for x in batch]).to(self.device)
        actions = torch.stack([torch.FloatTensor(x[1]) for x in batch]).to(self.device)
        rewards = torch.FloatTensor([x[2] for x in batch]).to(self.device)
        next_states = torch.stack([torch.FloatTensor(x[3]) for x in batch]).to(self.device)
        dones = torch.FloatTensor([x[4] for x in batch]).to(self.device)
        
        # Reshape actions to match network input
        actions = actions.view(self.batch_size, -1)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_actions = next_actions.view(self.batch_size, -1)
            target_q = self.target_critic(next_states, next_actions)
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actions_pred = self.actor(states)
        actions_pred = actions_pred.view(self.batch_size, -1)
        actor_loss = -self.critic(states, actions_pred).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self._soft_update(self.actor, self.target_actor)
        self._soft_update(self.critic, self.target_critic)
    
    def _soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, path):
        """Save the model weights."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """Load the model weights."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict']) 