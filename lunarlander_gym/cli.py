"""Console script for lunarlander_gym."""
import argparse
import sys
import gymnasium as gym
import numpy as np
import logging
def main():
    """Console script for lunarlander_gym."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--method",
        help="""Specifies the Reinforcement Agent method {
        0 -> Random,
        1 -> Gradient based optimization,
        2 -> Value estimation,
        3 -> Actor-critic
        }""",
        type=int,
        metavar='M',
        required=True,
    )
    args = parser.parse_args()
    env = gym.make("LunarLander-v2", render_mode="human")
    observation, info = env.reset(seed=42)
    
    
    if args.method > 3 or args.method < 0: 
        raise ValueError(f"input value {args.method} is an invalud method")
    if args.method == 0:
        logging.info(f'Random agent selected method {args.method}')
        policy = lambda x: env.action_space.sample()
    elif args.method == 1:
        raise NotImplemented(f'Method {args.method} is not supported')
    elif args.method == 2:
        raise NotImplemented(f'Method {args.method} is not supported')
    elif args.method == 3:
        raise NotImplemented(f'Method {args.method} is not supported')
    for _ in range(1000):
        action = policy(observation)  # User-defined policy function
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
    env.close()
        
    return 0


if __name__ == "__main__":
    main()
    sys.exit()  # pragma: no cover
