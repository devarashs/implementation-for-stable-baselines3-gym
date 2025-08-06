# Work Report: CartPole-v1 Environment

## Project Overview

This work report details the implementation, training, and evaluation of a Proximal Policy Optimization (PPO) agent for solving the CartPole-v1 environment from Gymnasium.

## Environment Description

**CartPole-v1** is a classic control problem where an agent must balance a pole on a cart by applying forces to move the cart left or right. It's one of the fundamental benchmark environments in reinforcement learning.

### Key Characteristics

- **Observation Space**: 4-dimensional continuous state space
  - `cart_position`: Horizontal position of the cart (-2.4 to +2.4)
  - `cart_velocity`: Velocity of the cart
  - `pole_angle`: Angle of the pole from vertical (-12° to +12°)
  - `pole_angular_velocity`: Angular velocity of the pole
- **Action Space**: 2 discrete actions controlling cart movement
  - `Action 0`: Push cart to the left (force = -10N)
  - `Action 1`: Push cart to the right (force = +10N)
- **Goal**: Keep the pole balanced (within ±12 degrees) for as long as possible
- **Episode Termination**: Pole falls beyond ±12°, cart moves beyond ±2.4 units, or 500 steps completed

### Reward Function Design

**Dense Reward Structure**:

- **Survival Reward**: +1 for each timestep the pole remains upright
- **No Goal Bonus**: Episode continues until failure or time limit
- **Maximum Episode Reward**: 500 points (if pole balanced for full episode)

**Reasoning Behind Dense Rewards**:

- Provides immediate feedback for good balance behavior
- Natural curriculum learning (longer balance = higher reward)
- Encourages stable, consistent policies
- Makes learning faster compared to sparse reward alternatives

## Algorithm Selection: Proximal Policy Optimization (PPO)

**Rationale**: PPO was chosen for this environment because:

- Policy gradient methods excel in environments with clear action preferences
- PPO's clipped objective prevents large policy updates, ensuring stable learning
- Well-suited for continuous state spaces with discrete actions
- Sample efficient for relatively simple control tasks

## Implementation Details

### Training Configuration

```python
# Hyperparameters (using SB3 defaults optimized for CartPole)
total_timesteps = 10,000
policy = "MlpPolicy"
verbose = 1
tensorboard_log = "cartpole-scripts/logs"
```

### Network Architecture

- **Policy**: MlpPolicy (Multi-Layer Perceptron)
- **Framework**: Stable Baselines3 PPO implementation
- **Backend**: PyTorch
- **Default Architecture**: 64x64 hidden layers with tanh activation

### PPO Loss Function

**Clipped Surrogate Objective**:

```python
# PPO loss components (implemented in Stable Baselines3)
ratio = π_new(a|s) / π_old(a|s)  # Probability ratio
clipped_ratio = clip(ratio, 1-ε, 1+ε)  # ε = 0.2 typically
policy_loss = -min(ratio * advantage, clipped_ratio * advantage)

# Value function loss
value_loss = MSE(V(s), returns)

# Entropy bonus for exploration
entropy_loss = -entropy_coef * entropy(π(s))

# Total loss
total_loss = policy_loss + value_loss_coef * value_loss + entropy_loss
```

**Key Components**:

- **Clipped Objective**: Prevents large policy updates (ε = 0.2)
- **Advantage Estimation**: Uses Generalized Advantage Estimation (GAE)
- **Value Function**: Shared network predicts state values
- **Entropy Regularization**: Encourages exploration

**Design Reasoning**:

- **Trust Region**: Clipping ensures conservative policy updates
- **Shared Network**: Policy and value function share feature extraction
- **Advantage Normalization**: Reduces variance in policy gradients

### Training Infrastructure

- **Monitoring**: Gymnasium Monitor for episode statistics
- **Evaluation**: EvalCallback every 10,000 steps
- **Logging**: TensorBoard integration for training metrics
- **Model Saving**: Best model saved based on evaluation performance

## Key Features Implemented

### 1. Policy Gradient Learning

- Direct policy optimization with advantage estimation
- Clipped surrogate objective for stable updates

### 2. Value Function Estimation

- Shared network architecture for policy and value function
- Generalized Advantage Estimation (GAE) for variance reduction

### 3. Trust Region Optimization

- Clipped probability ratios prevent destructive policy updates
- Maintains learning stability throughout training

### 4. Evaluation Protocol

- Separate evaluation environment
- Deterministic policy evaluation
- Best model saved automatically

## Training Process

### Data Flow

1. **Environment Setup**: CartPole-v1 wrapped with Monitor
2. **Agent Initialization**: PPO with MLP policy
3. **Training Loop**: 10,000 timesteps with periodic evaluation
4. **Model Persistence**: Final model saved as `ppo_cartpole.zip`

### Monitoring and Logging

- Episode rewards and lengths tracked via Monitor
- TensorBoard logs for policy loss, value loss, and entropy
- Evaluation metrics logged every 10k steps
- Best performing model automatically saved

## File Structure

```
cartpole-scripts/
├── train.py          # Training script with PPO implementation
├── play.py           # Inference script for trained model
├── logs/             # Training logs and TensorBoard data
└── __pycache__/      # Python bytecode cache

models/
├── ppo_cartpole.zip  # Final trained model
└── cartpole/
    └── best/         # Best model during training
```

## Results and Performance

### Training Characteristics

- **Total Training Time**: 10,000 timesteps
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Evaluation Frequency**: Every 10,000 steps
- **Final Model**: Saved as `ppo_cartpole.zip`

### Model Performance

The trained PPO agent demonstrates the ability to:

- Learn optimal pole balancing strategies quickly
- Achieve consistent performance near the maximum episode length (500 steps)
- Maintain stable policy throughout training
- Generalize to unseen initial conditions

## Technical Challenges Addressed

### 1. Rapid Convergence Requirements

- **Challenge**: CartPole can be solved quickly but requires careful tuning
- **Solution**: PPO's conservative updates prevent overshooting optimal policy

### 2. Sample Efficiency

- **Challenge**: Need to learn effective policy with limited samples
- **Solution**: On-policy learning with advantage estimation improves sample usage

### 3. Policy Stability

- **Challenge**: Maintaining performance once optimal policy is found
- **Solution**: PPO's clipped objective prevents policy degradation

## Code Quality and Best Practices

### 1. Modular Design

- Separate training and inference scripts
- Clear separation of concerns
- Environment monitoring integration

### 2. Monitoring and Evaluation

- Comprehensive logging setup
- Automatic best model saving
- TensorBoard integration for visualization

### 3. Reproducibility

- Stable Baselines3 framework ensures reproducible results
- Deterministic evaluation protocol
- Clear documentation of training procedure

## Usage Instructions

### Training

```bash
python -m cartpole-scripts.train
```

### Playing/Inference

```bash
python -m cartpole-scripts.play
```

### TensorBoard Monitoring

```bash
cd cartpole-scripts
tensorboard --logdir ./logs
```

## Dependencies

- **gymnasium**: Environment framework
- **stable-baselines3**: RL algorithms implementation
- **numpy**: Numerical computations
- **torch**: Deep learning backend (via SB3)

## Future Improvements

1. **Hyperparameter Optimization**: Systematic tuning of learning rate, batch size
2. **Advanced PPO Features**: Entropy regularization tuning, learning rate scheduling
3. **Network Architecture**: Experiment with different hidden layer configurations
4. **Multi-Environment Training**: Parallel environment training for faster convergence
5. **Transfer Learning**: Use CartPole as stepping stone to more complex environments

## Conclusion

The CartPole-v1 implementation successfully demonstrates the application of PPO to a fundamental control problem. The agent learns to balance the pole effectively within minimal training time, showcasing the efficiency of policy gradient methods for simple control tasks. The implementation serves as an excellent foundation for understanding PPO and reinforcement learning concepts, with comprehensive monitoring and evaluation infrastructure ensuring reliable training outcomes.
