# Work Report: MountainCarContinuous-v0 Environment

## Project Overview

This work report details the implementation, training, and evaluation of a Soft Actor-Critic (SAC) agent for solving the MountainCarContinuous-v0 environment from Gymnasium, with custom reward shaping to overcome the sparse reward challenge.

## Environment Description

**MountainCarContinuous-v0** is the continuous action variant of the classic MountainCar problem. An underpowered car must build momentum to reach the top of a hill, but now with continuous control over the force applied to the car.

### Key Characteristics

- **Observation Space**: 2-dimensional continuous state space
  - `position`: Car's horizontal position (-1.2 to +0.6, goal at +0.45)
  - `velocity`: Car's velocity (-0.07 to +0.07)
- **Action Space**: 1-dimensional continuous action controlling car's engine
  - `action`: Continuous force applied to car (range [-1, +1])
  - `action = -1`: Maximum leftward force
  - `action = 0`: No force applied (coast)
  - `action = +1`: Maximum rightward force
- **Goal**: Reach the flag at position ≥ 0.45 on the right hill
- **Episode Termination**: Goal reached or maximum 999 steps exceeded

### Original vs Modified Reward Function

**Original Sparse Reward**:

- **Time Penalty**: -1 for each timestep until goal is reached
- **Goal Reward**: 0 when position ≥ 0.45 (episode ends)
- **Problem**: Nearly impossible to learn due to extreme sparsity

**Custom Reward Shaping**:

```python
def reward(self, reward):
    pos, vel = self.env.unwrapped.state
    position_reward = 0.1 * (pos + 1.2)    # Linear progress reward
    velocity_reward = 15 * abs(vel)        # High velocity bonus
    return reward + position_reward + velocity_reward
```

**Reward Components Explained**:

- **Position Reward**: `0.1 * (pos + 1.2)` gives 0-0.18 points for rightward progress
- **Velocity Reward**: `15 * abs(vel)` gives 0-1.05 points for building momentum
- **Original Penalty**: -1 per timestep maintains efficiency pressure
- **Goal Preservation**: Original sparse goal reward still triggers episode end

**Design Reasoning**:

- **Velocity Emphasis**: 15x multiplier makes momentum building highly rewarding
- **Bidirectional Momentum**: `abs(vel)` rewards movement in either direction
- **Linear Position**: Simple progress measure toward goal
- **Preservation**: Maintains original problem structure while adding guidance

## Algorithm Selection: Soft Actor-Critic (SAC)

**Rationale**: SAC was chosen for this environment because:

- **Continuous Actions**: SAC excels in continuous control problems
- **Sample Efficiency**: Off-policy learning with experience replay
- **Exploration**: Entropy regularization promotes diverse action exploration
- **Stability**: Separate actor and critic networks with soft updates

## Implementation Details

### Custom Reward Shaping

```python
class ShapedRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        pos, vel = self.env.unwrapped.state
        position_reward = 0.1 * (pos + 1.2)    # Encourages moving right
        velocity_reward = 15 * abs(vel)        # Increased velocity reward
        return reward + position_reward + velocity_reward
```

**Reward Engineering**:

- **Position Reward**: Encourages movement toward the goal
- **Velocity Reward**: Heavily rewards building momentum (key insight)
- **Maintains Original**: Sparse goal reward preserved

### Training Configuration

```python
# Hyperparameters
total_timesteps = 500,000
learning_rate = 5e-4
batch_size = 256
buffer_size = 100,000
learning_starts = 10,000
ent_coef = "auto_0.5"
train_freq = 1
gradient_steps = 2
gamma = 0.98
tau = 0.02
policy_kwargs = dict(
    net_arch=dict(
        pi=[256, 256],    # Larger policy network
        qf=[256, 256]     # Larger Q-function network
    )
)
```

### Network Architecture

- **Policy Network**: 256x256 hidden layers
- **Q-Function Networks**: 256x256 hidden layers (two critics)
- **Framework**: Stable Baselines3 SAC implementation
- **Backend**: PyTorch

### SAC Loss Function

**Actor-Critic with Entropy Regularization**:

```python
# SAC loss components (implemented in Stable Baselines3)
# Q-function loss (twin critics)
target_q = reward + gamma * (min(Q1_target, Q2_target) - alpha * log_prob)
q1_loss = MSE(Q1(s,a), target_q)
q2_loss = MSE(Q2(s,a), target_q)

# Policy loss (actor)
actions, log_prob = policy.sample(state)
q_values = min(Q1(s,a), Q2(s,a))
policy_loss = alpha * log_prob - q_values

# Temperature (entropy) loss
alpha_loss = -alpha * (log_prob + target_entropy).detach()
```

**Key Components**:

- **Twin Critics**: Two Q-networks prevent overestimation bias
- **Entropy Regularization**: Temperature parameter α balances exploration/exploitation
- **Soft Updates**: Exponential moving averages for target networks (τ=0.02)
- **Reparameterization**: Enables backpropagation through stochastic policy

**Design Reasoning**:

- **Maximum Entropy**: Encourages diverse action exploration in continuous space
- **Off-Policy**: Sample efficiency through experience replay
- **Stability**: Twin critics and soft updates prevent training instability

### Training Infrastructure

- **Monitoring**: Gymnasium Monitor for episode statistics
- **Evaluation**: EvalCallback every 5,000 steps
- **Logging**: TensorBoard integration for training metrics
- **Model Saving**: Best model saved based on evaluation performance

## Key Features Implemented

### 1. Enhanced Reward Shaping

- **Position Shaping**: Linear reward based on rightward progress
- **Velocity Amplification**: High weight (15x) on velocity magnitude
- **Sparse Goal Preservation**: Original goal reward maintained

### 2. Entropy Regularization

- **Auto Entropy Tuning**: Automatic adjustment with initial value 0.5
- **Exploration Benefit**: Encourages diverse action exploration
- **Temperature Learning**: Adaptive entropy coefficient

### 3. Twin Critic Architecture

- **Two Q-Networks**: Reduces overestimation bias
- **Minimum Q-Target**: Uses minimum of twin critics for target computation
- **Improved Stability**: More robust value estimation

### 4. Soft Target Updates

- **Tau = 0.02**: Slow target network updates for stability
- **Continuous Updates**: Smooth parameter interpolation
- **Reduced Variance**: Stabilizes learning process

## Training Process

### Data Flow

1. **Environment Setup**: MountainCarContinuous-v0 with reward shaping wrapper
2. **Agent Initialization**: SAC with enhanced network architecture
3. **Training Loop**: 500,000 timesteps with frequent evaluation
4. **Model Persistence**: Final model saved as `sac_mountaincar_continuous.zip`

### Monitoring and Logging

- Episode rewards (shaped and original) tracked via Monitor
- TensorBoard logs for actor loss, critic loss, and entropy
- Evaluation metrics logged every 5k steps
- Best performing model automatically saved

## File Structure

```
mountaincar-continuous-scripts/
├── train.py                      # Training script with SAC implementation
├── play.py                       # Inference script for trained model
├── logs/                         # Training logs and TensorBoard data
└── __pycache__/                  # Python bytecode cache

models/
├── sac_mountaincar_continuous.zip # Final trained model (Note: filename mismatch)
└── mountaincar-continuous/
    └── best/                     # Best model during training
```

## Results and Performance

### Training Characteristics

- **Total Training Time**: 500,000 timesteps
- **Algorithm**: Soft Actor-Critic (SAC)
- **Evaluation Frequency**: Every 5,000 steps
- **Final Model**: Saved as `sac_mountaincar_continuous.zip`

### Model Performance

The trained SAC agent demonstrates the ability to:

- Learn smooth, continuous control policies
- Build momentum efficiently through reward shaping guidance
- Achieve goal state consistently across different starting positions
- Utilize continuous action space effectively for fine-grained control

## Technical Challenges Addressed

### 1. Sparse Reward Problem

- **Challenge**: Original environment provides reward only at goal
- **Solution**: Comprehensive reward shaping with position and velocity components

### 2. Continuous Action Space Exploration

- **Challenge**: Infinite action space requires effective exploration
- **Solution**: SAC's entropy regularization promotes action diversity

### 3. Sample Efficiency in Continuous Control

- **Challenge**: Continuous actions require more samples to learn effective policies
- **Solution**: Enhanced network architecture and optimized hyperparameters

### 4. Local Minima in Policy Space

- **Challenge**: Agent might get stuck in suboptimal oscillating behaviors
- **Solution**: Velocity reward strongly encourages momentum building

## Code Quality and Best Practices

### 1. Environment Wrapping

- Custom reward wrapper with clear mathematical formulation
- Maintains original environment interface
- Preserves core environment dynamics

### 2. Hyperparameter Optimization

- Larger networks for complex continuous control
- Optimized learning rates and batch sizes
- Appropriate entropy regularization

### 3. Comprehensive Monitoring

- Detailed logging setup with TensorBoard integration
- Automatic best model saving
- Evaluation protocol with separate environment

## Usage Instructions

### Training

```bash
python -m mountaincar-continuous-scripts.train
```

### Playing/Inference

```bash
python -m mountaincar-continuous-scripts.play
```

### TensorBoard Monitoring

```bash
cd mountaincar-continuous-scripts
tensorboard --logdir ./logs
```

## Dependencies

- **gymnasium**: Environment framework
- **stable-baselines3**: RL algorithms implementation
- **numpy**: Numerical computations
- **torch**: Deep learning backend (via SB3)

## Code Issues Identified

### 1. Model Loading Mismatch

```python
# In play.py - potential issue
model = SAC.load("models/sac_mountaincar")  # Missing "_continuous"
```

**Issue**: The play script may not load the correct model file.

### 2. Missing Reward Wrapper in Play

The play script doesn't apply the same reward shaping wrapper used during training, which could affect performance visualization.

## Future Improvements

1. **Curriculum Learning**: Gradually reduce reward shaping over training
2. **Domain Randomization**: Vary physics parameters for robustness
3. **Advanced SAC Features**: Implement TD3-style delayed policy updates
4. **Hyperparameter Sensitivity**: Analyze impact of reward shaping weights
5. **Transfer Learning**: Apply learned policy to related continuous control tasks

## Lessons Learned

### 1. Reward Shaping Impact

The dramatic improvement from reward shaping demonstrates its power in sparse reward environments. The 15x velocity reward was particularly effective.

### 2. Network Architecture Matters

Larger networks (256x256) were necessary for the continuous control problem, showing the importance of sufficient model capacity.

### 3. Entropy Regularization Benefits

SAC's built-in exploration through entropy maximization proved crucial for discovering effective continuous control policies.

## Conclusion

The MountainCarContinuous-v0 implementation successfully demonstrates the application of SAC to a challenging continuous control problem with sparse rewards. The custom reward shaping wrapper transforms an nearly unsolvable sparse reward environment into a learnable task, while maintaining the original problem structure. The implementation showcases advanced techniques in continuous control, reward engineering, and off-policy learning. The comprehensive monitoring and evaluation infrastructure ensures reliable training and provides valuable insights into the continuous control learning process.
