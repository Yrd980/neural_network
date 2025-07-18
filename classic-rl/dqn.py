import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, 
                 lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 batch_size=64, memory_size=10000, target_update_freq=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.target_update_freq = target_update_freq

        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.learn_step_counter = 0

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_t)
            return torch.argmax(q_values, dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        # 如果记忆回放池数量不足，跳过更新
        if len(self.memory) < self.batch_size:
            return

        # 从记忆回放池中采样
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # 当前Q值
        q_values = self.q_network(states).gather(1, actions)

        # 下一个状态的最大Q值(使用target网络)
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(dim=1, keepdim=True)[0]

        # 目标Q值
        target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        # 计算损失
        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # epsilon衰减
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 更新target网络
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

def train_dqn(env_name='CartPole-v1', episodes=500):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            # 对于CartPole，如果掉落则惩罚
            reward = reward if not done else -10

            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            total_reward += reward

        print(f"Episode {ep+1}, total_reward: {total_reward}, epsilon: {agent.epsilon:.3f}")

    env.close()
    return agent

if __name__ == "__main__":
    trained_agent = train_dqn()
