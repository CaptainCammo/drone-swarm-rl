import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DroneSwarmNetwork(nn.Module):
    """Neural network for the DQN agent"""
    def __init__(self, state_size, action_size, num_drones):
        super(DroneSwarmNetwork, self).__init__()
        self.input_size = state_size * num_drones
        self.output_size = action_size * num_drones
        
        self.network = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class DroneSwarmDQN:
    def __init__(self, state_size=12, action_size=4, num_drones=3):
        self.state_size = state_size
        self.action_size = action_size
        self.num_drones = num_drones
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.update_target_frequency = 100
        self.batch_size = 32
        
        # Neural Networks
        self.model = DroneSwarmNetwork(state_size, action_size, num_drones).to(self.device)
        self.target_model = DroneSwarmNetwork(state_size, action_size, num_drones).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        self.step_count = 0
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        flat_state = self._flatten_state(state)
        flat_next_state = self._flatten_state(next_state)
        self.memory.append((flat_state, action, reward, flat_next_state, done))
    
    def _flatten_state(self, state):
        """Convert dict state to flat array"""
        flat_state = []
        for i in range(self.num_drones):
            flat_state.extend(state[f'drone_{i}'])
        return np.array(flat_state, dtype=np.float32)
    
    def _dict_to_tensor(self, action_dict):
        """Convert action dictionary to tensor"""
        actions = []
        for i in range(self.num_drones):
            actions.extend(action_dict[f'drone_{i}'])
        return torch.FloatTensor(actions)
    
    def _tensor_to_dict(self, action_tensor):
        """Convert action tensor to dictionary"""
        actions = {}
        action_array = action_tensor.cpu().numpy()
        for i in range(self.num_drones):
            start_idx = i * self.action_size
            end_idx = start_idx + self.action_size
            actions[f'drone_{i}'] = np.clip(
                action_array[start_idx:end_idx], 
                -1, 
                1
            ).astype(np.float32)
        return actions
    
    def act(self, state):
        """Return actions for all drones"""
        if random.random() <= self.epsilon:
            # Random actions for each drone
            return {
                f'drone_{i}': np.random.uniform(
                    low=[-1, -1, -1, -1], 
                    high=[1, 1, 1, 1], 
                    size=4
                ).astype(np.float32)
                for i in range(self.num_drones)
            }
        
        # Convert state to tensor
        flat_state = self._flatten_state(state)
        state_tensor = torch.FloatTensor(flat_state).unsqueeze(0).to(self.device)
        
        # Get action values from model
        self.model.eval()
        with torch.no_grad():
            act_values = self.model(state_tensor)
        self.model.train()
        
        # Convert to action dictionary
        return self._tensor_to_dict(act_values[0])
    
    def replay(self):
        """Train on batch from replay memory"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch data
        states = torch.FloatTensor(np.vstack([x[0] for x in minibatch])).to(self.device)
        actions = [x[1] for x in minibatch]
        rewards = torch.FloatTensor([x[2] for x in minibatch]).to(self.device)
        next_states = torch.FloatTensor(np.vstack([x[3] for x in minibatch])).to(self.device)
        dones = torch.FloatTensor([x[4] for x in minibatch]).to(self.device)
        
        # Convert actions to tensors
        action_tensors = torch.stack([self._dict_to_tensor(a) for a in actions]).to(self.device)
        
        # Get current Q values
        current_q_values = self.model(states)
        
        # Get next Q values from target model
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            max_next_q = next_q_values.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Update Q values for taken actions
        for i in range(self.batch_size):
            current_q_values[i] = target_q_values[i]
        
        # Compute loss and update weights
        self.optimizer.zero_grad()
        loss = self.criterion(current_q_values, target_q_values.unsqueeze(1))
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target model periodically
        self.step_count += 1
        if self.step_count % self.update_target_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())
    
    def save(self, path):
        """Save model weights"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """Load model weights"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon'] 