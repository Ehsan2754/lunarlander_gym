import gymnasium as gym
import numpy as np
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)
policy = lambda x: np.random.randint(0,4)
for _ in range(1000):
    action = policy(observation)  # User-defined policy function
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()
