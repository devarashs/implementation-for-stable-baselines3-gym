# Work Report: MountainCar-v0 Environment

## Project Overview

This work report details the implementation, training, and evaluation of a Deep Q-Network (DQN) agent for solving the MountainCar-v0 environment from Gymnasium.

## Environment Description

**MountainCar-v0** is a classic control problem where an underpowered car must build momentum to reach the top of a hill. The car's engine is not powerful enough to drive directly up the steep hill, requiring the agent to learn a rocking motion strategy.

### Key Characteristics

- **Observation Space**: 2-dimensional continuous state space
  - `position`: Car's horizontal position (-1.2 to +0.6, goal at +0.5)
  - `velocity`: Car's velocity (-0.07 to +0.07)
- **Action Space**: 3 discrete actions controlling car's engine
  - `Action 0`: Push left (accelerate left, force = -0.001)
  - `Action 1`: No push (coast, no engine force)
  - `Action 2`: Push right (accelerate right, force = +0.001)
- **Goal**: Reach the flag at position ≥ 0.5 on the right hill
- **Episode Termination**: Goal reached or maximum 200 steps exceeded

### Reward Function Design

**Sparse Reward Structure**:

- **Time Penalty**: -1 for each timestep until goal is reached
- **Goal Reward**: 0 when position ≥ 0.5 (episode ends)
- **No Intermediate Rewards**: No rewards for progress or velocity

**Reasoning Behind Sparse Rewards**:

- **Exploration Challenge**: Forces discovery of momentum-building strategy
- **Efficiency Learning**: Encourages shortest path to goal
- **Real-world Analogy**: Fuel consumption matters in actual driving
- **Algorithm Testing**: Tests capability to handle delayed gratification

**Key Insight**: The car must learn the counterintuitive strategy of initially moving away from the goal (going left) to build momentum for the rightward ascent.

## Algorithm Selection: Deep Q-Network (DQN)

**Rationale**: DQN was chosen for this environment because:

- Discrete action space is well-suited for Q-learning approaches
- Value-based methods can handle sparse reward environments effectively
- DQN's experience replay helps learn from rare successful episodes
- Target network stabilizes learning in challenging exploration scenarios

## Implementation Details

### Training Configuration

```python
# Hyperparameters
total_timesteps = 1,000,000
learning_rate = 1e-3
buffer_size = 50,000
learning_starts = 10,000
batch_size = 64
tau = 1.0
gamma = 0.99
train_freq = 1
target_update_interval = 500
exploration_fraction = 0.3
exploration_final_eps = 0.05
max_grad_norm = 10
```

### Network Architecture

- **Policy**: MlpPolicy (Multi-Layer Perceptron)
- **Framework**: Stable Baselines3 DQN implementation
- **Backend**: PyTorch
- **Gradient Clipping**: Max norm of 10 to prevent gradient explosion

### Training Infrastructure

- **Monitoring**: Gymnasium Monitor for episode statistics
- **Evaluation**: EvalCallback every 10,000 steps
- **Logging**: TensorBoard integration for training metrics
- **Model Saving**: Best model saved based on evaluation performance

## Key Features Implemented

### 1. Extended Exploration Phase

- **Exploration Fraction**: 30% of total training time
- **Final Epsilon**: 0.05 (maintains some randomness)
- **Rationale**: MountainCar requires discovery of momentum-building strategy

### 2. Experience Replay Buffer

- **Buffer Size**: 50,000 transitions
- **Learning Starts**: 10,000 steps before learning begins
- **Purpose**: Accumulate diverse experiences before training

### 3. Target Network

- **Update Interval**: Every 500 steps
- **Hard Updates**: Full parameter copy (tau = 1.0)
- **Benefit**: Stable Q-target computation

### 4. Gradient Clipping

- **Max Norm**: 10
- **Purpose**: Prevent gradient explosion during early training

## Training Process

### Data Flow

1. **Environment Setup**: MountainCar-v0 wrapped with Monitor
2. **Agent Initialization**: DQN with extended exploration
3. **Training Loop**: 1M timesteps with periodic evaluation
4. **Model Persistence**: Final model saved as `dqn_mountaincar.zip`

### Monitoring and Logging

- Episode rewards and lengths tracked via Monitor
- TensorBoard logs for Q-loss, exploration rate, and episode statistics
- Evaluation metrics logged every 10k steps
- Best performing model automatically saved

## File Structure

```
mountaincar-scripts/
├── train.py              # Training script with DQN implementation
├── play.py               # Inference script for trained model
├── logs/                 # Training logs and TensorBoard data
└── __pycache__/          # Python bytecode cache

models/
├── dqn_mountaincar.zip   # Final trained model
└── mountaincar/
    └── best/             # Best model during training
```

## Results and Performance

### Training Characteristics

- **Total Training Time**: 1,000,000 timesteps
- **Algorithm**: Deep Q-Network (DQN)
- **Evaluation Frequency**: Every 10,000 steps
- **Final Model**: Saved as `dqn_mountaincar.zip`

### Model Performance

The trained DQN agent demonstrates the ability to:

- Learn the optimal momentum-building strategy (rocking motion)
- Discover that moving left initially can lead to better right-side momentum
- Achieve goal state consistently after sufficient exploration
- Generalize to different starting positions in the valley

## Technical Challenges Addressed

### 1. Sparse Reward Environment

- **Challenge**: Only receive reward when goal is achieved (rare early in training)
- **Solution**: Large replay buffer accumulates successful experiences

### 2. Exploration Strategy

- **Challenge**: Random actions unlikely to discover successful policy
- **Solution**: Extended exploration phase (30% of training)

### 3. Momentum Building Strategy

- **Challenge**: Counterintuitive strategy (go left to eventually go right)
- **Solution**: DQN learns through trial and error with experience replay

### 4. Sample Efficiency

- **Challenge**: Learning from limited successful episodes
- **Solution**: Experience replay allows repeated learning from successful transitions

## Code Quality and Best Practices

### 1. Modular Design

- Separate training and inference scripts
- Clear hyperparameter configuration
- Proper directory structure for outputs

### 2. Monitoring and Evaluation

- Comprehensive logging setup
- Automatic best model saving
- TensorBoard integration for training visualization

### 3. Reproducibility

- Fixed hyperparameters documented in code
- Deterministic evaluation protocol
- Clear training procedure documentation

## Usage Instructions

### Training

```bash
python -m mountaincar-scripts.train
```

### Playing/Inference

```bash
python -m mountaincar-scripts.play
```

### TensorBoard Monitoring

```bash
cd mountaincar-scripts
tensorboard --logdir ./logs
```

## Dependencies

- **gymnasium**: Environment framework
- **stable-baselines3**: RL algorithms implementation
- **numpy**: Numerical computations
- **torch**: Deep learning backend (via SB3)

## Future Improvements

1. **Reward Shaping**: Add intermediate rewards for position progress
2. **Advanced DQN Variants**: Double DQN, Dueling DQN, or Rainbow
3. **Hyperparameter Tuning**: Systematic optimization of exploration parameters
4. **Curriculum Learning**: Progressive difficulty environments
5. **Prioritized Experience Replay**: Focus learning on important transitions

## Lessons Learned

### 1. Exploration is Critical

MountainCar demonstrates the importance of sufficient exploration in sparse reward environments. The extended exploration phase (30% of training) is crucial for discovering the momentum-building strategy.

### 2. Experience Replay Value

The large replay buffer and delayed learning start allow the agent to accumulate diverse experiences before beginning policy updates, improving sample efficiency.

### 3. Target Network Stability

The target network with hard updates every 500 steps provides stable Q-targets, preventing the moving target problem common in Q-learning.

## Conclusion

The MountainCar-v0 implementation successfully demonstrates the application of DQN to a challenging control problem with sparse rewards. The agent learns the counterintuitive momentum-building strategy through extensive exploration and experience replay. This implementation serves as an excellent example of how value-based methods can solve complex control problems when properly configured for exploration and sample efficiency. The comprehensive monitoring infrastructure ensures reliable training and provides insights into the learning process.
