# Drone Swarm Reinforcement Learning Environment

This project provides a customizable environment for training AI agents to control multiple drones in 3D space. The environment is built using the Gymnasium framework and provides a realistic simulation of drone dynamics with airplane-like aerodynamics.

## Features

- Support for multiple drones in a swarm
- 3D physics simulation with realistic airplane aerodynamics
- Boundary constraints with penalties for out-of-bounds movement
- Centroid-based swarm coordination
- Customizable reward function
- Full state observation including position, velocity, orientation, and angular velocity
- Action space for controlling thrust and angular rates

## State Space (per drone)

- Position (x, y, z)
- Velocity (vx, vy, vz)
- Orientation (roll, pitch, yaw)
- Angular velocity (wx, wy, wz)

## Action Space (per drone)

- Thrust (normalized between 0 and 1)
- Roll rate command (normalized between -1 and 1)
- Pitch rate command (normalized between -1 and 1)
- Yaw rate command (normalized between -1 and 1)

## Installation

1. Make sure you have Python 3.8+ installed
2. Install Poetry if you haven't already:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
3. Clone this repository
4. Install dependencies:
   ```bash
   poetry install
   ```

## Training the DDPG Agent

The project uses a Deep Deterministic Policy Gradient (DDPG) agent for training, which is well-suited for continuous action spaces like drone control. The training script `train_waypoint_DDPG_agent.py` provides a complete pipeline for training and testing the agent.

### Basic Training

To start training with default parameters:
```bash
python RL_Agents/train_waypoint_DDPG_agent.py
```

### Custom Training Parameters

You can customize the training process using command-line arguments:

```bash
python RL_Agents/train_waypoint_DDPG_agent.py \
    --episodes 500 \           # Number of training episodes
    --render \                 # Enable rendering during training
    --render_interval 5 \      # Render every 5 episodes
    --render_delay 0.02        # Delay between renders in seconds
```

### Testing a Trained Model

To test a trained model:
```bash
python RL_Agents/train_waypoint_DDPG_agent.py \
    --mode test \              # Test mode
    --test_episodes 10 \       # Number of test episodes
    --model_path models/DDPG_training_20240321_143022/ddpg_agent_final.pt  # Path to model
```

### Available Command-line Arguments

Training arguments:
- `--episodes`: Number of training episodes (default: 1000)
- `--mode`: Either 'train' or 'test' (default: 'train')
- `--render`: Flag to enable rendering during training
- `--render_delay`: Delay between renders in seconds (default: 0.01)
- `--render_interval`: How often to render (every N episodes) (default: 10)

Testing arguments:
- `--test_episodes`: Number of test episodes (default: 5)
- `--model_path`: Path to the model to test (default: 'models/DDPG_training_*/ddpg_agent_final.pt')

### Training Process

The training process:
1. Creates a scanning pattern of waypoints for the drones to follow
2. Initializes the DDPG agent with appropriate hyperparameters
3. Trains the agent through multiple episodes
4. Saves the model and metrics periodically
5. Generates plots of training metrics
6. Optionally tests the trained model

### Output Files

The training process generates:
- Model weights in the `models/` directory
- Training metrics in the `training_results/` directory
- Plots of training progress, including:
  - Episode rewards
  - Episode lengths
  - Waypoints reached
  - Formation quality rewards
  - Velocity alignment rewards
  - Waypoint rewards
  - Distance rewards
  - Wrong direction penalties
  - Termination reasons

### Environment Parameters

The environment is configured with the following parameters:
- Number of drones: 3
- Maximum steps per episode: 5000
- Waypoint size: 60.0
- Waypoint reward: 150.0
- Distance reward factor: 0.3
- Maximum steps away: 50
- Wrong direction penalty: 200.0

### DDPG Agent Hyperparameters

The DDPG agent uses the following hyperparameters:
- Actor learning rate: 0.001
- Critic learning rate: 0.001
- Discount factor (gamma): 0.99
- Soft target update parameter (tau): 0.001
- Batch size: 64
- Replay buffer size: 100000

## Notes

- The DDPG agent is used instead of DQN because the action space is continuous (thrust and angular rates)
- Training progress is automatically saved and can be resumed
- The environment supports both training and testing modes
- Rendering can be enabled/disabled to balance training speed and visualization

## Usage

Check out the example in `examples/basic_usage.py` for a simple demonstration of how to use the environment:
