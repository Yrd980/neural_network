import numpy as np


class ContextualBandit:
    def __init__(self, k=3, d=5):
        self.k = k
        self.d = d
        self.weights = np.random.randn(k, d)

    def get_context(self):
        return np.random.randn(self.d)

    def get_reward(self, action, context):
        mean_reward = np.dot(self.weights[action], context)
        return np.random.normal(mean_reward, 1.0)


class LinearUCB:
    def __init__(self, k, d, alpha=1.0):
        self.k = k
        self.d = d
        self.alpha = alpha
        self.A = [np.identity(d) for _ in range(k)]
        self.b = [np.zeros(d) for _ in range(k)]

    def select_action(self, context):
        p = np.zeros(self.k)
        for a in range(self.k):
            A_inv = np.linalg.inv(self.A[a])
            theta_a = A_inv.dot(self.b[a])
            p[a] = theta_a.dot(context) + self.alpha * np.sqrt(
                context.dot(A_inv).dot(context)
            )
        return np.argmax(p)

    def update(self, action, context, reward):
        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context


if __name__ == "__main__":
    np.random.seed(42)
    k = 3
    d = 5
    bandit = ContextualBandit(k, d)
    agent = LinearUCB(k, d, alpha=1.0)

    iterations = 1000
    rewards = np.zeros(iterations)

    for t in range(iterations):
        context = bandit.get_context()
        action = agent.select_action(context)
        reward = bandit.get_reward(action, context)
        agent.update(action, context, reward)
        rewards[t] = reward

    print("Average reward:", np.mean(rewards))
