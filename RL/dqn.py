#!/usr/bin/env python3
"""
Deep Q Network (DQN) implementation from scratch using PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
from collections import deque
from typing import List, Tuple

# Always seed the random number generator
random.seed(42)
np.random.seed(42)


class ReplayMemory:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dims: List[int] = [64, 64]
    ):
        super().__init__()
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.001,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 128,
        target_update_freq: int = 10,
        hidden_dims: List[int] = [64, 64],
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.training_step = 0

        self.q_network = QNetwork(state_dim, action_dim, hidden_dims)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dims)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayMemory(buffer_capacity)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()


def train_dqn(
    env, agent: DQNAgent, num_episodes: int, max_steps: int = 500, render_env=None
):
    rewards_history = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()

            total_reward += reward
            state = next_state

            if done:
                break

        rewards_history.append(total_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(
                f"Episode {episode + 1}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.3f}"
            )

        if render_env is not None and (episode + 1) % 10 == 0:
            print(f"\n--- Visualizing controller after {episode + 1} episodes ---")
            visualize_agent(render_env, agent, max_steps=max_steps)

    return rewards_history


def visualize_agent(env, agent: DQNAgent, max_steps: int = 500):
    state, _ = env.reset()
    total_reward = 0

    for step in range(max_steps):
        env.render()
        time.sleep(0.02)
        action = agent.select_action(state, training=False)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_reward += reward
        state = next_state

        if done:
            break

    time.sleep(0.5)
    env.render()
    print(f"Visualization complete. Reward: {total_reward}")


def smooth_rewards(rewards, weight=0.9):
    smoothed = []
    for r in rewards:
        if smoothed:
            smoothed.append(weight * smoothed[-1] + (1 - weight) * r)
        else:
            smoothed.append(r)
    return smoothed


def compute_variance(rewards, window=10):
    variances = []
    for i in range(len(rewards)):
        start = max(0, i - window + 1)
        window_rewards = rewards[start : i + 1]
        variances.append(np.var(window_rewards))
    return variances


def plot_rewards(rewards_history):
    import matplotlib.pyplot as plt

    smoothed = smooth_rewards(rewards_history)
    variances = compute_variance(rewards_history)

    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history, alpha=0.3, label="Raw Rewards")
    plt.plot(smoothed, label="Smoothed Rewards", linewidth=2)
    plt.fill_between(
        range(len(smoothed)),
        np.array(smoothed) - np.sqrt(variances),
        np.array(smoothed) + np.sqrt(variances),
        alpha=0.3,
        label="±1 Std Dev",
    )
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN Training Rewards")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    import gymnasium as gym

    VISUALIZE = True

    env = gym.make("CartPole-v1")
    render_env = gym.make("CartPole-v1", render_mode="human") if VISUALIZE else None

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        hidden_dims=[30, 30],
        epsilon_decay=0.999,
        epsilon_min=0.0001,
        buffer_capacity=10000,
        batch_size=64,
        target_update_freq=10,
    )

    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    print("Training DQN on CartPole-v1...")

    rewards = train_dqn(
        env, agent, num_episodes=200, max_steps=500, render_env=render_env
    )

    env.close()
    if render_env:
        render_env.close()
    print(f"Training complete. Final avg reward: {np.mean(rewards[-10:]):.2f}")

    plot_rewards(rewards)

    render_env = gym.make("CartPole-v1", render_mode="human")

    visualize_agent(render_env, agent)

    render_env.close()
