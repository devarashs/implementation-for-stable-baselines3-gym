# Usage Guide

This guide explains how to set up your environment and run each pipeline.

## 1. Create and Activate Conda Environment

```bash
conda create -n rl python=3.12.9
conda activate rl
```

## 2. Install Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

## 3. Train Pipelines

### 3.1 Acrobot-v1 with DQN

```bash
python -m acrobot-scripts.train
```

### 3.2 CartPole-v1 with PPO

```bash
python -m cartpole-scripts.train
```

### 3.3 MountainCar-v0 with DQN

```bash
python -m mountaincar-scripts.train
```

### 3.4 MountainCar-Continuous-v0 with SAC

```bash
python -m mountaincar-continuous-scripts.train
```

### 3.5 LunarLander-v3 with DQN

```bash
python -m lunarlander-scripts.train
```

## 4. Play Pipelines

### 4.1 Acrobot-v1 with DQN

```bash
python -m acrobot-scripts.play
```

### 4.2 CartPole-v1 with PPO

```bash
python -m cartpole-scripts.play
```

### 4.3 MountainCar-v0 with DQN

```bash
python -m mountaincar-scripts.play
```

### 4.4 MountainCar-Continuous-v0 with SAC

```bash
python -m mountaincar-continuous-scripts.play
```

### 4.5 LunarLander-v3 with DQN

```bash
python -m lunarlander-scripts.play
```
