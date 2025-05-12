# Reinforcement Learning Playground

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.6.0-brightgreen.svg)](https://stable-baselines3.readthedocs.io/en/master/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.1.1-yellow.svg)](https://gymnasium.farama.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

## Goal and Purpose

This repository is a collection of Reinforcement Learning (RL) experiments designed for educational purposes. It aims to provide practical examples of how popular RL algorithms perform in different environments. This project is ideal for anyone looking to:

- Learn about fundamental RL concepts.
- Experiment with different RL algorithms.
- Test RL models in various environments.

The primary goal is to offer a clear and accessible introduction to RL, showcasing how different algorithms can be applied to solve various control problems. It's a starting point for understanding the core mechanics of RL and exploring its potential.

## What's Inside

This project includes implementations of the following RL algorithms and environments:

- **CartPole-v1:** Solved using **Proximal Policy Optimization (PPO)**. PPO is a policy gradient algorithm that optimizes the policy directly.
- **MountainCar-v0:** Solved using **Deep Q-Network (DQN)**. DQN is a value-based algorithm that learns to estimate the optimal action-value function.
- **MountainCarContinuous-v0:** Solved using **Soft Actor-Critic (SAC)**. SAC is an off-policy actor-critic algorithm that aims to maximize both reward and entropy.
- **Acrobot-v1:** Solved using **Deep Q-Network (DQN)**. This demonstrates the application of DQN to a more complex control problem.
- **LunarLander-v3:** Solved using **Deep Q-Network (DQN)**. This example shows how DQN can be applied to a classic control problem with continuous state space and discrete actions.
