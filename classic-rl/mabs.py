import numpy as np
import matplotlib.pyplot as plt


class EpsilonGreedyBandit:
    def __init__(self, k=10, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon
        self.q_true = np.random.randn(k)
        self.q_estimate = np.zeros(k)
        self.action_count = np.zeros(k)

    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.q_estimate)

    def step(self, action):
        reward = np.random.randn() + self.q_true[action]
        self.action_count[action] += 1
        alpha = 1.0 / self.action_count[action]
        self.q_estimate[action] += alpha * (reward - self.q_estimate[action])
        return reward


def run_bandit(k=10, epsilon=0.1, steps=1000, runs=2000):
    rewards = np.zeros((runs, steps))
    for r in range(runs):
        bandit = EpsilonGreedyBandit(k, epsilon)
        for t in range(steps):
            action = bandit.act()
            reward = bandit.step(action)
            rewards[r, t] = reward
    avg_rewards = np.mean(rewards, axis=0)
    return avg_rewards


if __name__ == "__main__":
    epsilons = [0.0, 0.1, 0.01]
    for eps in epsilons:
        avg_rewards = run_bandit(epsilon=eps)
        plt.plot(avg_rewards, label=f"epsilon={eps}")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.show()
