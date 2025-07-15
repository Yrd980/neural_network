import pygame
import numpy as np
import random
import math
from collections import defaultdict
import matplotlib.pyplot as plt


class Particle:
    def __init__(self, x, y, color, lifetime=30):
        self.x = x
        self.y = y
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-2, 2)
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.size = random.uniform(2, 6)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifetime -= 1
        self.vx *= 0.98
        self.vy *= 0.98

    def draw(self, screen):
        if self.lifetime > 0:
            alpha = int(255 * (self.lifetime / self.max_lifetime))
            color = (*self.color, alpha)
            size = int(self.size * (self.lifetime / self.max_lifetime))
            if size > 0:
                surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (size, size), size)
                screen.blit(surf, (self.x - size, self.y - size))


class Trail:
    def __init__(self, max_length=20):
        self.positions = []
        self.max_length = max_length

    def add_position(self, x, y):
        self.positions.append((x, y))
        if len(self.positions) > self.max_length:
            self.positions.pop(0)

    def draw(self, screen):
        if len(self.positions) > 1:
            for i, (x, y) in enumerate(self.positions):
                alpha = int(255 * (i / len(self.positions)))
                size = int(8 * (i / len(self.positions)))
                if size > 0:
                    color = (255, 100, 100, alpha)
                    surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                    pygame.draw.circle(surf, color, (size, size), size)
                    screen.blit(surf, (x - size, y - size))


class SimpleGameEnv:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Enhanced RL Game - Visual Experience")

        self.player_pos = [self.width // 2, self.height // 2]
        self.target_pos = [self.width // 2, self.height // 2]
        self.done = False
        self.clock = pygame.time.Clock()

        self.particles = []
        self.trail = Trail()
        self.target_pulse = 0
        self.player_glow = 0
        self.background_stars = self.generate_stars(100)
        self.success_animation = 0
        self.steps_taken = 0

        self.bg_color = (10, 10, 30)
        self.player_color = (255, 50, 50)
        self.target_color = (50, 255, 50)
        self.trail_color = (255, 100, 100)

        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

    def generate_stars(self, count):
        stars = []
        for _ in range(count):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            brightness = random.randint(50, 200)
            size = random.randint(1, 3)
            stars.append((x, y, brightness, size))
        return stars

    def reset(self):
        self.player_pos = [
            random.randint(50, self.width - 50),
            random.randint(50, self.height - 50),
        ]
        self.target_pos = [self.width // 2, self.height // 2]
        self.done = False
        self.particles = []
        self.trail = Trail()
        self.success_animation = 0
        self.steps_taken = 0
        return self.get_state()

    def get_state(self):
        return np.array(self.player_pos, dtype=np.float32)

    def discretize_state(self, state, grid_size=40):
        x = int(state[0] // grid_size)
        y = int(state[1] // grid_size)
        return (x, y)

    def step(self, action):
        self.steps_taken += 1

        old_pos = self.player_pos.copy()

        move_speed = 8
        if action == 0:
            self.player_pos[1] -= move_speed
        elif action == 1:
            self.player_pos[1] += move_speed
        elif action == 2:
            self.player_pos[0] -= move_speed
        elif action == 3:
            self.player_pos[0] += move_speed

        self.player_pos[0] = np.clip(self.player_pos[0], 20, self.width - 20)
        self.player_pos[1] = np.clip(self.player_pos[1], 20, self.height - 20)

        if old_pos != self.player_pos:
            for _ in range(3):
                self.particles.append(
                    Particle(
                        old_pos[0] + random.randint(-5, 5),
                        old_pos[1] + random.randint(-5, 5),
                        (255, 150, 150),
                        lifetime=20,
                    )
                )

        self.trail.add_position(self.player_pos[0], self.player_pos[1])

        center = np.array(self.target_pos)
        player = np.array(self.player_pos)
        dist = np.linalg.norm(center - player)
        reward = -dist / 200  # Normalize reward

        if dist < 30:
            self.done = True
            reward += 100
            self.success_animation = 60  

            for _ in range(50):
                self.particles.append(
                    Particle(
                        self.player_pos[0] + random.randint(-20, 20),
                        self.player_pos[1] + random.randint(-20, 20),
                        (255, 255, 100),
                        lifetime=60,
                    )
                )

        return self.get_state(), reward, self.done, {}

    def update_effects(self):
        self.particles = [p for p in self.particles if p.lifetime > 0]
        for particle in self.particles:
            particle.update()

        self.target_pulse += 0.1
        self.player_glow = (self.player_glow + 0.15) % (2 * math.pi)

        if self.success_animation > 0:
            self.success_animation -= 1

    def draw_background(self):
        for y in range(self.height):
            color_intensity = int(10 + 20 * (y / self.height))
            color = (color_intensity, color_intensity, color_intensity + 10)
            pygame.draw.line(self.screen, color, (0, y), (self.width, y))

        for x, y, brightness, size in self.background_stars:
            twinkle = int(
                brightness + 30 * math.sin(pygame.time.get_ticks() * 0.01 + x * 0.1)
            )
            twinkle = max(0, min(255, twinkle))
            color = (twinkle, twinkle, twinkle)
            pygame.draw.circle(self.screen, color, (x, y), size)

    def draw_target(self):
        pulse_size = 25 + 10 * math.sin(self.target_pulse)
        glow_size = pulse_size + 15

        glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(
            glow_surf, (0, 255, 0, 50), (glow_size, glow_size), glow_size
        )
        self.screen.blit(
            glow_surf, (self.target_pos[0] - glow_size, self.target_pos[1] - glow_size)
        )

        for i in range(3):
            ring_size = pulse_size - i * 8
            if ring_size > 0:
                pygame.draw.circle(
                    self.screen, self.target_color, self.target_pos, ring_size, 2
                )

        pygame.draw.circle(self.screen, (255, 255, 255), self.target_pos, 5)

    def draw_player(self):
        glow_intensity = 50 + 30 * math.sin(self.player_glow)
        glow_surf = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (255, 50, 50, int(glow_intensity)), (20, 20), 20)
        self.screen.blit(glow_surf, (self.player_pos[0] - 20, self.player_pos[1] - 20))

        pygame.draw.circle(
            self.screen,
            self.player_color,
            (int(self.player_pos[0]), int(self.player_pos[1])),
            12,
        )

        pygame.draw.circle(
            self.screen,
            (255, 150, 150),
            (int(self.player_pos[0] - 3), int(self.player_pos[1] - 3)),
            6,
        )

    def draw_ui(self):
        dist = np.linalg.norm(np.array(self.target_pos) - np.array(self.player_pos))
        dist_text = self.small_font.render(
            f"Distance: {dist:.1f}", True, (255, 255, 255)
        )
        self.screen.blit(dist_text, (10, 10))

        steps_text = self.small_font.render(
            f"Steps: {self.steps_taken}", True, (255, 255, 255)
        )
        self.screen.blit(steps_text, (10, 40))

        if self.success_animation > 0:
            success_text = self.font.render("SUCCESS!", True, (255, 255, 100))
            text_rect = success_text.get_rect(
                center=(self.width // 2, self.height // 2 - 100)
            )
            self.screen.blit(success_text, text_rect)

    def render(self):
        self.update_effects()

        self.draw_background()

        self.trail.draw(self.screen)

        for particle in self.particles:
            particle.draw(self.screen)

        self.draw_target()

        self.draw_player()

        self.draw_ui()

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()


class QLearningAgent:
    def __init__(
        self,
        actions=4,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = defaultdict(lambda: np.zeros(actions))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.actions - 1)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]

        if done:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state])

        self.q_table[state][action] = current_q + self.learning_rate * (
            target_q - current_q
        )

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_agent(episodes=1000, render_frequency=50):
    env = SimpleGameEnv()
    agent = QLearningAgent()

    scores = []
    episode_lengths = []

    for episode in range(episodes):
        state = env.reset()
        discrete_state = env.discretize_state(state)
        total_reward = 0
        steps = 0
        max_steps = 1000

        render_this_episode = episode % render_frequency == 0

        while not env.done and steps < max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return scores, episode_lengths
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        render_this_episode = not render_this_episode

            action = agent.choose_action(discrete_state)
            next_state, reward, done, _ = env.step(action)
            next_discrete_state = env.discretize_state(next_state)

            agent.learn(discrete_state, action, reward, next_discrete_state, done)

            discrete_state = next_discrete_state
            total_reward += reward
            steps += 1

            if render_this_episode:
                env.render()

        scores.append(total_reward)
        episode_lengths.append(steps)

        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(
                f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}"
            )

    env.close()
    return scores, episode_lengths, agent


def test_agent(agent, episodes=10):
    env = SimpleGameEnv()

    for episode in range(episodes):
        state = env.reset()
        discrete_state = env.discretize_state(state)
        total_reward = 0
        steps = 0

        print(f"Testing Episode {episode + 1}")

        while not env.done and steps < 1000:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        state = env.reset()
                        discrete_state = env.discretize_state(state)
                        total_reward = 0
                        steps = 0

            action = np.argmax(agent.q_table[discrete_state])
            state, reward, done, _ = env.step(action)
            discrete_state = env.discretize_state(state)

            total_reward += reward
            steps += 1

            env.render()

        print(
            f"Episode {episode + 1} completed in {steps} steps with reward {total_reward:.2f}"
        )

        pygame.time.wait(1000)

    env.close()


def plot_training_results(scores, episode_lengths):
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(scores, alpha=0.6, color="lightblue", linewidth=0.5)
    if len(scores) > 100:
        moving_avg = np.convolve(scores, np.ones(100) / 100, mode="valid")
        ax1.plot(
            range(99, len(scores)),
            moving_avg,
            color="orange",
            linewidth=2,
            label="Moving Average",
        )
    ax1.set_title("Training Scores", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Score")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(episode_lengths, color="lightgreen", alpha=0.7)
    if len(episode_lengths) > 100:
        moving_avg = np.convolve(episode_lengths, np.ones(100) / 100, mode="valid")
        ax2.plot(
            range(99, len(episode_lengths)),
            moving_avg,
            color="red",
            linewidth=2,
            label="Moving Average",
        )
    ax2.set_title("Episode Lengths", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("🎮 Enhanced Visual RL Game Starting...")
    print("Controls during training:")
    print("  - SPACE: Toggle rendering for current episode")
    print("  - ESC/Close: Exit training")
    print("Controls during testing:")
    print("  - R: Reset current episode")
    print("  - ESC/Close: Exit testing")

    print("\n🤖 Training Q-Learning Agent...")
    scores, episode_lengths, trained_agent = train_agent(
        episodes=1000, render_frequency=100
    )

    print("\n✅ Training completed!")
    print(f"Final average score: {np.mean(scores[-100:]):.2f}")

    plot_training_results(scores, episode_lengths)

    print("\n🧪 Testing trained agent with full visual experience...")
    test_agent(trained_agent, episodes=5)
