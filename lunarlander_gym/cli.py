"""Console script for lunarlander_gym."""
import argparse
import sys
import gymnasium as gym
import numpy as np
import logging
from lunarlander_gym.agents import *
# from agents import *

def main():
    """Console script for lunarlander_gym."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--method",
        help="""Specifies the Reinforcement Agent method {
        0 -> Random,
        1 -> Gradient Policy Agent,
        2 -> Q-Learning Agent,
        3 -> Actor-critic Agent
        }""",
        type=int,
        metavar='M',
        required=True,
    )
    args = parser.parse_args()


    if args.method > 3 or args.method < 0: 
        raise ValueError(f"input value {args.method} is an invalud method")
    if args.method == 0:
        agent = RandomAgent(rendergif=True)
        print(agent)
        agent.test()
    elif args.method == 1:
        # agent = VanillaPolicyGradientAgent(render=False)
        # print(agent)
        # agent.train(episode=10_000,save_every=1000)
        agent = VanillaPolicyGradientAgent(rendergif=True)
        print(agent)
        agent.test(model='./output/VanillaPolicyGradientAgent/train_outputs/trainVanillaPolicyGradientAgentEpisode=10000Reward=49.10.pth')
    elif args.method == 2:
        # agent = QLearningAget(render=False)
        # print(agent)
        # agent.train(episode=3000,save_every=250)
        agent = QLearningAget(rendergif=True)
        print(agent)
        agent.test(model='./output/QLearningAget/train_outputs/trainQLearningAgetEpisode=3000Reward=184.91.pth')

  
    elif args.method == 3:
        # agent = ActorCriticAgent(render=False)
        # print(agent)
        # agent.train(episode=3000,save_every=250)
        agent = ActorCriticAgent(rendergif=True)
        print(agent)
        agent.test(model='./output/ActorCriticAgent/train_outputs/trainActorCriticAgentEpisode=3000Reward=284.60.pth')

        
    return 0


if __name__ == "__main__":
    main()
    sys.exit()  # pragma: no cover
