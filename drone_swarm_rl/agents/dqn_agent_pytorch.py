import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class SwarmQNetwork(nn.Module):
    """Neural network for approximating Q-values for the entire swarm."""
    
    def __init__(self, state_size, action_size, num_drones, hidden_size=128):
        """Initialize the Swarm Q-Network.
        
        Args:
            state_size: Dimension of each drone's state
            action_size: Dimension of each drone's action
            num_drones: Number of drones in the swarm
            hidden_size: Size of hidden layers
        """
        super(SwarmQNetwork, self).__init__()
        
        # Total input size is state_size * num_drones
        self.input_size = state_size * num_drones
        
        # Total output size is action_size * num_drones
        self.output_size = action_size * num_drones
        
        # Define network architecture
        self.model = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.output_size)
        )
    
    def forward(self, state):
        """Forward pass through the network."""
        return self.model(state)

class DroneSwarmDQN:
    """DQN Agent for controlling a swarm of drones with a single network."""
    
    def __init__(self, state_size, action_size, num_drones, 
                 memory_size=10000, batch_size=64, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 learning_rate=0.001, device=None):
        """Initialize the DQN Agent.
        
        Args:
            state_size: Dimension of each state for a single drone
            action_size: Dimension of each action for a single drone
            num_drones: Number of drones in the swarm
            memory_size: Size of the replay memory
            batch_size: Size of the training batch
            gamma: Discount factor
            epsilon: Exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            learning_rate: Learning rate for the optimizer
            device: Device to run the model on (cpu or cuda)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_drones = num_drones
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        
        # Set device (CPU or GPU)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Initialize replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Create a single Q-network for the entire swarm
        self.q_network = SwarmQNetwork(state_size, action_size, num_drones).to(self.device)
        
        # Create target Q-network
        self.target_network = SwarmQNetwork(state_size, action_size, num_drones).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Set to evaluation mode
        
        # Create optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Counter for updating target network
        self.update_counter = 0
        self.target_update_freq = 10  # Update target network every 10 steps
    
    def _flatten_state(self, state_dict):
        """Convert state dictionary to flat tensor."""
        flat_state = []
        for i in range(self.num_drones):
            drone_id = f'drone_{i}'
            flat_state.extend(state_dict[drone_id])
        return np.array(flat_state, dtype=np.float32)
    
    def _unflatten_action(self, action_flat):
        """Convert flat action array to dictionary."""
        actions = {}
        for i in range(self.num_drones):
            drone_id = f'drone_{i}'
            start_idx = i * self.action_size
            end_idx = start_idx + self.action_size
            
            # Extract and process this drone's actions
            drone_action = action_flat[start_idx:end_idx].copy()
            
            # Ensure thrust is between 0 and 1
            drone_action[0] = np.clip(drone_action[0], 0, 1)
            
            # Ensure other actions are between -1 and 1
            drone_action[1:] = np.clip(drone_action[1:], -1, 1)
            
            actions[drone_id] = drone_action.astype(np.float32)
        return actions
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        # Flatten states for storage
        flat_state = self._flatten_state(state)
        flat_next_state = self._flatten_state(next_state)
        
        # Store flattened experience
        self.memory.append((flat_state, action, reward, flat_next_state, done))
    
    def act(self, state, use_epsilon=True):
        """Choose actions for all drones based on the current state.
        
        Args:
            state: Dictionary of states for each drone
            use_epsilon: Whether to use epsilon-greedy policy
            
        Returns:
            Dictionary of actions for each drone
        """
        # Flatten the state dictionary
        flat_state = self._flatten_state(state)
        
        # Epsilon-greedy action selection
        if use_epsilon and np.random.rand() <= self.epsilon:
            # Random actions for all drones
            flat_actions = np.random.uniform(low=-1, high=1, size=self.action_size * self.num_drones)
            # Ensure thrust values are between 0 and 1
            for i in range(self.num_drones):
                thrust_idx = i * self.action_size
                flat_actions[thrust_idx] = np.clip((flat_actions[thrust_idx] + 1) / 2, 0, 1)
        else:
            # Get actions from Q-network
            state_tensor = torch.FloatTensor(flat_state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            
            # Convert Q-values to actions
            q_values = q_values.cpu().numpy()[0]
            
            # Process Q-values to get actions for each drone
            flat_actions = np.zeros(self.action_size * self.num_drones)
            
            for i in range(self.num_drones):
                start_idx = i * self.action_size
                end_idx = start_idx + self.action_size
                
                # Get Q-values for this drone
                drone_q = q_values[start_idx:end_idx]
                
                # Normalize Q-values to the range [-1, 1]
                q_min, q_max = np.min(drone_q), np.max(drone_q)
                if q_max > q_min:  # Avoid division by zero
                    normalized_q = 2 * (drone_q - q_min) / (q_max - q_min) - 1
                else:
                    normalized_q = np.zeros_like(drone_q)
                
                # Ensure thrust is between 0 and 1
                normalized_q[0] = (normalized_q[0] + 1) / 2  # Map from [-1, 1] to [0, 1]
                
                # Store normalized actions
                flat_actions[start_idx:end_idx] = normalized_q
        
        # Convert flat actions to dictionary
        return self._unflatten_action(flat_actions)
    
    def replay(self):
        """Train the agent by sampling from replay memory."""
        # Skip if not enough samples in memory
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Extract batch data
        states = np.array([sample[0] for sample in minibatch])
        actions_dict = [sample[1] for sample in minibatch]
        rewards = np.array([sample[2] for sample in minibatch])
        next_states = np.array([sample[3] for sample in minibatch])
        dones = np.array([sample[4] for sample in minibatch])
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        # Get current Q-values
        current_q = self.q_network(states_tensor)
        
        # Get next Q-values from target network
        with torch.no_grad():
            next_q = self.target_network(next_states_tensor)
            
            # For each drone, get the maximum Q-value for the next state
            target_q = current_q.clone()
            
            for i in range(self.batch_size):
                # For each sample in the batch
                for d in range(self.num_drones):
                    # For each drone
                    start_idx = d * self.action_size
                    end_idx = start_idx + self.action_size
                    
                    # Get the maximum Q-value for this drone's next state
                    max_next_q = torch.max(next_q[i, start_idx:end_idx])
                    
                    # Calculate target Q-value
                    target_value = rewards_tensor[i] + (1 - dones_tensor[i]) * self.gamma * max_next_q
                    
                    # Update all Q-values for this drone
                    target_q[i, start_idx:end_idx] = target_value
        
        # Compute loss
        loss = self.criterion(current_q, target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network with weights from main network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath):
        """Save model weights to file."""
        save_dict = {
            'q_network': self.q_network.state_dict(),
            'hyperparams': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'num_drones': self.num_drones,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay
            }
        }
        torch.save(save_dict, filepath)
    
    def load(self, filepath):
        """Load model weights from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load hyperparameters if available
        if 'hyperparams' in checkpoint:
            hyperparams = checkpoint['hyperparams']
            self.gamma = hyperparams.get('gamma', self.gamma)
            self.epsilon = hyperparams.get('epsilon', self.epsilon)
            self.epsilon_min = hyperparams.get('epsilon_min', self.epsilon_min)
            self.epsilon_decay = hyperparams.get('epsilon_decay', self.epsilon_decay)
        
        # Load network weights
        if 'q_network' in checkpoint:
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['q_network']) 