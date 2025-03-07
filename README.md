# Drone Swarm Reinforcement Learning Environment

This project provides a customizable environment for training AI agents to control multiple drones in 3D space. The environment is built using the Gymnasium framework and provides a realistic simulation of drone dynamics.

## Features

- Support for multiple drones in a swarm
- 3D physics simulation with gravity and thrust
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

## Usage

Check out the example in `examples/basic_usage.py` for a simple demonstration of how to use the environment:

```python
from drone_swarm_rl.environment import DroneSwarmEnv

# Create environment with 3 drones
env = DroneSwarmEnv(num_drones=3)

# Reset the environment
obs, _ = env.reset()

# Run simulation
for _ in range(100):
    action = {drone_id: env.action_space[drone_id].sample() 
             for drone_id in env.action_space.spaces}
    obs, reward, terminated, truncated, info = env.step(action)
```

## Customization

You can customize various aspects of the environment:

- Number of drones
- Physical parameters (mass, max thrust, etc.)
- Reward function
- Initial conditions
- Maximum episode length
