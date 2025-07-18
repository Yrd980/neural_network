import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)

def select_action(policy_net, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy_net(state).squeeze(0)
    action = torch.multinomial(probs, 1).item()
    return action, probs[action]

def train_reinforce(env_name='CartPole-v1', 
                    gamma=0.99, 
                    lr=1e-3, 
                    episodes=2000):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    for episode in range(episodes):
        log_probs = []
        rewards = []

        state = env.reset()
        done = False
        while not done:
            action, prob = select_action(policy_net, state)
            log_probs.append(torch.log(prob))

            next_state, reward, done, info = env.step(action)
            rewards.append(reward)
            state = next_state

        # 计算回报并更新参数
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns)
        # 归一化处理（可选），减小方差
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        loss = 0
        for log_prob, Gt in zip(log_probs, returns):
            loss += -log_prob * Gt

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (episode+1) % 100 == 0:
            print(f"Episode {episode+1}, total reward = {sum(rewards)}")

    env.close()
    return policy_net

if __name__ == "__main__":
    trained_policy = train_reinforce()
