---
## 🧠 `GOAL.md` — MiniHack-Room-5x5-v0 Agent Objective

### 🎮 Environment Overview: `MiniHack-Room-5x5-v0`

MiniHack is a reinforcement learning environment built on top of the classic rogue-like game *NetHack*. Unlike simple Gym environments, MiniHack introduces elements like procedural generation, partial observability, inventory management, and symbolic reasoning.

The `MiniHack-Room-5x5-v0` environment is one of the simplest MiniHack maps and is used for introductory experimentation. It provides a compact, deterministic 5x5 room grid.
---

### 🎯 Agent's Objective

- The agent starts at a **random position** inside a 5x5 room.
- It must **reach the staircase (`>` symbol)**, which serves as the **goal/exit point**.
- The environment terminates once the agent reaches the goal or exceeds the maximum number of steps (usually 50–100).
- The agent receives:

  - **Positive reward** for reaching the goal.
  - **Sparse or no intermediate reward**, depending on configuration.

---

### 🕹️ Controls and Action Space

- The agent has access to **NetHack-compatible discrete actions**, such as:

  - Move in 8 directions (up, down, left, right, and diagonals)
  - Wait or perform noop actions

- The observation includes:

  - **Symbolic ASCII representation** of the map (e.g., `@`, `.`, `>`)
  - Optional: inventory, message log, agent status (HP, position, etc.)

---

### 🧪 Reinforcement Learning Setup

- The observation and reward space is **partially observable** and **sparse**.
- Suitable algorithms:

  - **PPO** (Proximal Policy Optimization) — recommended baseline
  - DQN and SAC can be tried, but discrete action space fits PPO better

- The agent must **learn spatial awareness and planning** without any pre-programmed map.

---

### ✅ Success Criteria

A successful agent will:

- Consistently **reach the exit (goal)** across multiple randomized runs.
- Do so in **fewer steps over time**, showing signs of exploration and exploitation.

---
