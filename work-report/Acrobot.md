# Work Report: Acrobot-v1 Environment

## Project Overview

This work report details the implementation, training, and evaluation of a Deep Q-Network (DQN) agent for solving the Acrobot-v1 environment from Gymnasium.

## Environment Description

**Acrobot-v1** is a classic control problem where a two-link pendulum (acrobot) must swing up to reach a target height. The acrobot consists of two links connected by joints, with only the second joint being actuated.

### Key Characteristics

- **Observation Space**: 6-dimensional continuous state space
  - `cos(θ1), sin(θ1)`: First joint angle components
  - `cos(θ2), sin(θ2)`: Second joint angle components
  - `θ1_dot, θ2_dot`: Angular velocities of both joints
- **Action Space**: 3 discrete actions controlling the second joint actuator
  - `Action 0`: Apply -1 torque (counterclockwise force)
  - `Action 1`: Apply 0 torque (no actuation)
  - `Action 2`: Apply +1 torque (clockwise force)
- **Goal**: Raise the end of the acrobot to a height of at least 1 unit above the base
- **Episode Termination**: When goal is reached or maximum 500 steps exceeded

### Reward Function Design

**Original Gymnasium Reward Structure**:

- **Sparse Reward**: -1 for each timestep until goal is reached
- **Goal Reward**: 0 when the tip height ≥ 1.0

**Reasoning Behind Sparse Rewards**:

- Forces the agent to learn efficient swing-up strategies
- Encourages reaching the goal as quickly as possible
- Mirrors real-world scenarios where energy efficiency matters
- Tests the algorithm's ability to handle delayed gratification

## Algorithm Selection: Deep Q-Network (DQN)

**Rationale**: DQN was chosen for this environment because:

- Discrete action space fits well with Q-learning approaches
- Value-based methods excel in environments with sparse rewards
- DQN's experience replay helps with sample efficiency in control problems

## Implementation Details

### Training Configuration

```python
# Hyperparameters
total_timesteps = 1,000,000
learning_rate = 1e-3
buffer_size = 50,000
learning_starts = 5,000
batch_size = 64
gamma = 0.99
train_freq = 1
target_update_interval = 500
exploration_fraction = 0.3
exploration_final_eps = 0.05
```

### Network Architecture

- **Policy**: MlpPolicy (Multi-Layer Perceptron)
- **Framework**: Stable Baselines3 DQN implementation
- **Backend**: PyTorch

### DQN Loss Function

**Temporal Difference Loss**:

```python
# Q-learning loss (implemented in Stable Baselines3)
target_q = reward + gamma * max(Q_target(next_state)) * (1 - done)
current_q = Q_network(state)[action]
loss = MSE(current_q, target_q)
```

**Key Components**:

- **Target Q-Values**: Computed using separate target network for stability
- **Bellman Equation**: Q(s,a) = r + γ \* max(Q(s',a'))
- **Mean Squared Error**: Minimizes difference between predicted and target Q-values
- **Experience Replay**: Samples random batches to break temporal correlations

**Design Reasoning**:

- **Target Network**: Prevents moving target problem during training
- **MSE Loss**: Smooth gradient flow for continuous Q-value updates
- **Clipped Gradients**: Prevents exploding gradients in early training phases

### Training Infrastructure

- **Monitoring**: Gymnasium Monitor for episode statistics
- **Evaluation**: EvalCallback every 10,000 steps
- **Logging**: TensorBoard integration for training metrics
- **Model Saving**: Best model saved based on evaluation performance

## Key Features Implemented

### 1. Experience Replay Buffer

- Buffer size: 50,000 transitions
- Enables off-policy learning and breaks temporal correlations

### 2. Target Network

- Separate target network updated every 500 steps
- Improves training stability by reducing moving target problem

### 3. Epsilon-Greedy Exploration

- Linear decay from 1.0 to 0.05 over 30% of training
- Balances exploration vs exploitation

### 4. Evaluation Protocol

- Separate evaluation environment
- Deterministic policy evaluation
- Best model saved automatically

## Training Process

### Data Flow

1. **Environment Setup**: Acrobot-v1 wrapped with Monitor
2. **Agent Initialization**: DQN with MLP policy
3. **Training Loop**: 1M timesteps with periodic evaluation
4. **Model Persistence**: Final model saved as `dqn_acrobot.zip`

### Monitoring and Logging

- Episode rewards tracked via Monitor
- TensorBoard logs for loss, Q-values, and exploration rate
- Evaluation metrics logged every 10k steps
- Best performing model automatically saved

## File Structure

```
acrobot-scripts/
├── train.py          # Training script with DQN implementation
├── play.py           # Inference script for trained model
├── logs/             # Training logs and TensorBoard data
└── __pycache__/      # Python bytecode cache

models/
├── dqn_acrobot.zip   # Final trained model
└── acrobot/
    └── best/         # Best model during training
```

## Results and Performance

### Training Characteristics

- **Total Training Time**: 1,000,000 timesteps
- **Algorithm**: Deep Q-Network (DQN)
- **Evaluation Frequency**: Every 10,000 steps
- **Final Model**: Saved as `dqn_acrobot.zip`

### Model Performance

The trained DQN agent demonstrates the ability to:

- Learn the pendulum dynamics through trial and error
- Develop swing-up strategies to reach the target height
- Achieve goal states consistently after sufficient training

## Technical Challenges Addressed

### 1. Sparse Reward Problem

- **Challenge**: Only receive reward when goal is achieved
- **Solution**: DQN's experience replay helps learn from successful episodes

### 2. Continuous State Space

- **Challenge**: 6D continuous observation space
- **Solution**: MLP network approximates Q-function over continuous states

### 3. Exploration vs Exploitation

- **Challenge**: Need to explore enough to find successful strategies
- **Solution**: Epsilon-greedy with extended exploration period (30% of training)

## Code Quality and Best Practices

### 1. Modular Design

- Separate training and inference scripts
- Clear separation of concerns
- Configurable hyperparameters

### 2. Monitoring and Evaluation

- Comprehensive logging setup
- Automatic best model saving
- TensorBoard integration for visualization

### 3. Reproducibility

- Fixed random seeds through Stable Baselines3
- Deterministic evaluation protocol
- Clear documentation of hyperparameters

## Usage Instructions

### Training

```bash
python -m acrobot-scripts.train
```

### Playing/Inference

```bash
python -m acrobot-scripts.play
```

### TensorBoard Monitoring

```bash
cd acrobot-scripts
tensorboard --logdir ./logs
```

## Dependencies

- **gymnasium**: Environment framework
- **stable-baselines3**: RL algorithms implementation
- **numpy**: Numerical computations
- **torch**: Deep learning backend (via SB3)

## Future Improvements

1. **Hyperparameter Tuning**: Systematic optimization of learning rate, buffer size
2. **Advanced DQN Variants**: Double DQN, Dueling DQN, Rainbow DQN
3. **Reward Shaping**: Potential-based reward modifications to speed up learning
4. **Network Architecture**: Experiment with different hidden layer sizes
5. **Curriculum Learning**: Progressive difficulty training

## Conclusion

The Acrobot-v1 implementation successfully demonstrates the application of DQN to a classic control problem. The agent learns to swing up the double pendulum through trial and error, showcasing the effectiveness of value-based reinforcement learning in environments with discrete actions and sparse rewards. The comprehensive monitoring and evaluation infrastructure ensures reliable training and performance assessment.
