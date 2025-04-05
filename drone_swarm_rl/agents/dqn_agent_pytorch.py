import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os

class SwarmDQNNetwork(nn.Module):
    def __init__(self, num_drones, state_size_per_drone, action_size_per_drone):
        super(SwarmDQNNetwork, self).__init__()
        
        self.num_drones = num_drones
        self.state_size_per_drone = state_size_per_drone
        self.action_size_per_drone = action_size_per_drone
        
        # Calculate total input size (all drone states concatenated)
        total_input_size = num_drones * state_size_per_drone
        
        # First hidden layer - processes combined state
        self.layer1 = nn.Sequential(
            nn.Linear(total_input_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )
        
        # Second hidden layer
        self.layer2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )
        
        # Third hidden layer
        self.layer3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        
        # Fourth hidden layer
        self.layer4 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        
        # Fifth hidden layer
        self.layer5 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        # Output layer - produces actions for all drones
        self.output_layer = nn.Linear(128, num_drones * action_size_per_drone)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: [batch_size, num_drones * state_size]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.output_layer(x)
        
        # Reshape output to [batch_size, num_drones, action_size]
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_drones, self.action_size_per_drone)
        
        return x

class DroneSwarmDQN:
    def __init__(self, observation_space, action_space, learning_rate=0.001, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 batch_size=64, target_update=10, memory_size=100000):
        """
        Initialize the DQN agent.
        
        Args:
            observation_space: Gym observation space
            action_space: Gym action space
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor
            epsilon_start: Starting value of epsilon
            epsilon_end: Minimum value of epsilon
            epsilon_decay: Decay rate for epsilon
            batch_size: Size of training batches
            target_update: Frequency of target network updates
            memory_size: Size of replay memory
        """
        # Get dimensions from the first drone (assuming all drones have same state/action sizes)
        first_drone_id = next(iter(observation_space.spaces.keys()))
        state_size_per_drone = observation_space.spaces[first_drone_id].shape[0]
        action_size_per_drone = action_space.spaces[first_drone_id].shape[0]
        num_drones = len(observation_space.spaces)
        
        # Initialize networks
        self.policy_net = SwarmDQNNetwork(num_drones, state_size_per_drone, action_size_per_drone)
        self.target_net = SwarmDQNNetwork(num_drones, state_size_per_drone, action_size_per_drone)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Initialize replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Initialize hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_counter = 0
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
    
    def act(self, state, use_epsilon=True):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state tensor
            use_epsilon: Whether to use epsilon-greedy exploration
        
        Returns:
            Action tensor
        """
        if use_epsilon and random.random() < self.epsilon:
            # Random action
            action = torch.randn(state.size(0), self.policy_net.num_drones, self.policy_net.action_size_per_drone)
            return action.clamp(-1, 1)  # Clamp to [-1, 1] range
        
        # Greedy action
        with torch.no_grad():
            state = state.to(self.device)
            action = self.policy_net(state)
            return action
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """
        Train the network using experience replay.
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Compute current Q-values (with gradients)
        current_q_values = self.policy_net(states)
        
        # Get action indices for gather operation
        with torch.no_grad():
            action_indices = torch.argmax(current_q_values, dim=2, keepdim=True)
        
        # Select Q-values for taken actions
        current_q_values = current_q_values.gather(2, action_indices)
        
        # Compute next state Q-values (without gradients)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(2)[0]
        
        # Reshape rewards to match next_q_values shape [batch_size, num_drones]
        rewards = rewards.unsqueeze(1).expand(-1, self.policy_net.num_drones)
        dones = dones.unsqueeze(1).expand(-1, self.policy_net.num_drones)
        
        # Compute the expected Q values
        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute Huber loss
        loss = nn.SmoothL1Loss()(current_q_values.squeeze(), expected_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path):
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """
        Load the model from a file.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']