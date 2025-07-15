import gymnasium as gym
import numpy as np


def q_learning_frozen_lake(
    env_name="FrozenLake-v1",
    alpha=0.5,
    gamma=0.95,
    epsilon=1.0,
    episodes=20000,
    epsilon_decay=0.9995,
    min_epsilon=0.05,
    verbose=True,
):
    env = gym.make(env_name, is_slippery=True)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    rewards_history = []
    success_count = 0

    goal_state = 15
    hole_states = {5, 7, 11, 12}

    def get_shaped_reward(state, next_state, reward, done):
        shaped_reward = reward

        shaped_reward -= 0.01

        def manhattan_distance(s1, s2):
            r1, c1 = s1 // 4, s1 % 4
            r2, c2 = s2 // 4, s2 % 4
            return abs(r1 - r2) + abs(c1 - c2)

        if not done:
            old_dist = manhattan_distance(state, goal_state)
            new_dist = manhattan_distance(next_state, goal_state)
            shaped_reward += 0.1 * (old_dist - new_dist)

        return shaped_reward

    for ep in range(episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 100:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state] + np.random.randn(n_actions) * 0.001)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            shaped_reward = get_shaped_reward(state, next_state, reward, done)

            Q[state, action] += alpha * (
                shaped_reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

            state = next_state
            total_reward += reward
            steps += 1

        rewards_history.append(total_reward)
        if total_reward > 0:
            success_count += 1

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if verbose and (ep + 1) % 2000 == 0:
            recent_success_rate = sum(rewards_history[-2000:]) / 2000
            print(
                f"Episode {ep + 1}: Success rate (last 2000): {recent_success_rate:.3f}, Epsilon: {epsilon:.3f}"
            )

    env.close()
    return Q, rewards_history, success_count


def test_random_agent(episodes=1000):
    env = gym.make("FrozenLake-v1", is_slippery=True)
    successes = 0

    for _ in range(episodes):
        state, info = env.reset()
        done = False
        steps = 0

        while not done and steps < 100:
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

            if reward > 0:
                successes += 1
                break

    env.close()
    return successes / episodes


if __name__ == "__main__":
    print("Testing random agent baseline...")
    random_success_rate = test_random_agent()
    print(f"Random agent success rate: {random_success_rate:.3f}")

    print("\nTraining Q-learning agent on FrozenLake...")
    Q, rewards, success_count = q_learning_frozen_lake()

    print("\nTraining Results:")
    print(f"Total successes: {success_count}")
    print(f"Success rate over last 2000 episodes: {np.mean(rewards[-2000:]):.3f}")

    print("\nNon-zero Q-values:")
    for state in range(len(Q)):
        for action in range(len(Q[state])):
            if abs(Q[state, action]) > 0.001:
                action_names = ["Left", "Down", "Right", "Up"]
                print(
                    f"State {state}, Action {action_names[action]}: {Q[state, action]:.3f}"
                )

    print("\nPreferred actions by state:")
    action_names = ["Left", "Down", "Right", "Up"]
    for state in range(len(Q)):
        best_action = np.argmax(Q[state])
        best_q = Q[state, best_action]
        if best_q != 0:
            print(f"State {state}: {action_names[best_action]} (Q-value: {best_q:.3f})")

    print("\nTesting trained agent (100 episodes):")
    env = gym.make("FrozenLake-v1", is_slippery=True)
    test_successes = 0

    for test_ep in range(100):
        state, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 100:
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        if total_reward > 0:
            test_successes += 1

    print(f"Test success rate: {test_successes}/100 = {test_successes/100:.1%}")
    print(f"Improvement over random: {(test_successes/100 - random_success_rate):.3f}")
    env.close()

    print("\n" + "=" * 50)
    print("Testing on non-slippery FrozenLake for comparison:")

    def q_learning_non_slippery():
        env = gym.make("FrozenLake-v1", is_slippery=False)
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        Q = np.zeros((n_states, n_actions))

        for ep in range(1000):
            state, info = env.reset()
            done = False

            while not done:
                if np.random.rand() < 0.1:  
                    action = env.action_space.sample()
                else:
                    action = np.argmax(Q[state])

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                Q[state, action] += 0.1 * (
                    reward + 0.99 * np.max(Q[next_state]) - Q[state, action]
                )

                state = next_state

        env.close()
        return Q

    Q_non_slip = q_learning_non_slippery()
    print("Non-slippery Q-values (should be non-zero):")
    for state in range(len(Q_non_slip)):
        for action in range(len(Q_non_slip[state])):
            if abs(Q_non_slip[state, action]) > 0.001:
                action_names = ["Left", "Down", "Right", "Up"]
                print(
                    f"State {state}, Action {action_names[action]}: {Q_non_slip[state, action]:.3f}"
                )
