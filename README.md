# RL DQN for Super Mario Bros

A Deep Q-Network (DQN) agent trained to play Super Mario Bros using PyTorch.

## Features

* Deep Q Network (DQN)
* Experience Replay
* Target Network Synchronization
* Epsilon-Greedy Exploration
* Frame Stacking
* Grayscale Observation
* Observation Resizing (84x84)

## Project Structure

* `train.py` - Training script
* `test.py` - Evaluation script
* `agent.py` - DQN agent implementation
* `dqn.py` - Neural network architecture
* `experience_replay.py` - Replay memory buffer
* `mario_env.py` - Mario environment wrappers
* `parameters.yaml` - Hyperparameters

## Technologies Used

* Python
* PyTorch
* Gym Super Mario Bros
* NES-Py
* NumPy

## Training

```bash
python train.py
```

## Testing

```bash
python test.py
```

## Results

The agent learns to move through the Mario environment using reinforcement learning and improves its reward over training episodes.


