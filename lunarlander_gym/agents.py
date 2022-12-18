"""
lunarlander_gym Agents
"""
import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import os
from models import *
import logging


class BaseAgent:
    """Base Agent class
    defines the interface to possible agents"""

    def __init__(
        self,
        render=True,
        rendergif=False,
        path="./output",
        name="BaseAgent",
        seed=543,
    ):
        """
        Parameters:
        -----------
        render: boolean, whether to render the environment or not
        rendergif: boolean, whether to render the environment result in a gif file
        path: string, path to save the results
        name: str
            name of the agent
        seed: int
            random seed
        """
        self.name = name
        self.env = (
            gym.make("LunarLander-v2", render_mode="rgb_array")
            if rendergif
            else gym.make("LunarLander-v2", render_mode="human")
        )
        if not render:
            self.env = gym.make("LunarLander-v2")
        self.render = render
        self.rendergif = rendergif
        self.path = os.path.join(path, self.name)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.train_path = os.path.join(self.path, "train_outputs")
        if not os.path.exists(self.train_path):
            os.makedirs(self.train_path)
        self.test_path = os.path.join(self.path, "test_outputs")
        if not os.path.exists(self.test_path):
            os.makedirs(self.test_path)

        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(self.seed)

    def __str__(self) -> str:
        return f"""Name: {self.name},
        Environment: {self.env}
        Render: {self.render},
        Train Path: {self.train_path},
        Test Path: {self.test_path}
        """

    def reset(self):
        """
        Reset the agent
        """
        raise NotImplementedError

    def choose_action(self, observation):
        """
        Step the agent
        Parameters:
        -----------
        observation: np.array
            action to take
        """
        raise NotImplementedError

    def train(self):
        """
        Learn the environment by agent
        """
        raise NotImplementedError

    def test(self, model, gif=False):
        """
        Test the agent

        Parameters:
        -----------
        model: str
            path to saved model "
        gif: str|bool
            save the output model as gif to the given path, if None, it will be as humanmode.
        """
        raise NotImplementedError

    def save(self, path):
        """
        Save the agent

        Parameters:
        -----------
        path: str
            path to save the agent"""
        raise NotImplementedError

    def saveFramesToGif(self, frames, path="./", filename="result.gif"):
        plt.title(filename[:-4])
        print('\t',filename[:-4])
        plt.figure(
            figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72
        )
        patch = plt.imshow(frames[0])
        plt.axis("off")

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(
            plt.gcf(), animate, frames=len(frames), interval=50
        )
        anim.save(os.path.join(path, filename), writer="imagemagick", fps=60)


"""
Random Agent
"""


class RandomAgent(BaseAgent):
    def __init__(
        self,
        render=True,
        rendergif=False,
        path="./output",
        name="RandomAgent",
        seed=543,
    ):
        super().__init__(render, rendergif, path, name, seed)

    def choose_action(self):
        return self.env.action_space.sample()

    def train(self):
        raise NotImplementedError("Random Agent does not support training")

    def test(self, episode=12, max_frames=0, render_every=4):
        """        
        Parameters:
        -----------
        model: str
            path to saved model, empty string tests the current model on the agent"
        episode: int
            number of episodes
        max_frames: int
            maximum number of frames to test the agent in each episode
        render_every: int
            render the environment every @param render_every episodes"""

        observation, info = self.env.reset(seed=self.seed)
        cur_reward = 0
        for e in range(1,episode+1):
            observation, info = self.env.reset(seed=self.seed)
            done = False
            frame = 0
            frames = []
            while not done:
                frame += 1

                action = self.choose_action()
                observation, reward, terminated, truncated, info = self.env.step(action)
                cur_reward += reward

                done = terminated or truncated
                if max_frames and (frame > max_frames):
                    done = True
                if self.render and not (e % render_every):
                    if self.rendergif:
                        frames.append(self.env.render())
                    else:
                        self.env.render()
            if self.rendergif and not (e % render_every):
                self.saveFramesToGif(
                    frames,
                    self.test_path,
                    f"Test{self.name}Episodes{e}Reward={cur_reward:.2f}.gif",
                )

            print(f"Test{self.name}\tEpisode={e}\tReward={cur_reward:.2f}")


"""        
Policy Gradient Agent
"""


class PolicyGradientAgent(BaseAgent):
    """
    Policy Gradient Reinforcement Agent
    """

    def __init__(self, env, name="PolicyGradientAgent", lr=1e-1, reward_decay=9.5e-1):
        """
        Parameters:
        -----------
        """
        super().__init__(env, name)
        self.action_space = np.arange(env.action_space.start, env.action_space.n)
        self.observation_space = env.observation_space
        self.lr = lr
        self.reward_decay = reward_decay

        self.memory = []


class ActorCriticAgent(BaseAgent):
    """
    Actor Critic Reinforcement Agent
    """

    def __init__(
        self,
        render=True,
        rendergif=False,
        path="./output",
        name="ActorCriticAgent",
        lr=0.2e-1,
        gamma=9.9e-1,
        betas=(0.9, 0.999),
        seed=543,
    ):
        super().__init__(render, rendergif, path, name, seed)
        self.env.reset(seed=self.seed)

        self.lr = lr
        self.gamma = gamma
        self.betas = betas
        self.policy = ActorCriticModule()
        self.optimizer = optim.Adam(
            self.policy.parameters(), lr=self.lr, betas=self.betas
        )

    def reset(self):
        """
        Reset the agent
        """
        self.env.reset(seed=self.seed)
        self.policy = ActorCriticModule()
        self.optimizer = optim.Adam(
            self.policy.parameters(), lr=self.lr, betas=self.betas
        )

    def choose_action(self, observation):
        """
        Step the agent
        Parameters:
        -----------
        observation: np.array of shape (n_samples,...)
            state vector of the environment
        """
        return self.policy(observation)

    def train(self, episode=10_000, save_every=500, render_every=1000, max_frames=0):
        """
        Learn the environment by agent
        Parameters:
        -----------
        episode: int
            number of episodes
        save_every: bool |int
            saves the model weights every @param save_every episodes
        render_every: int
            render the environment every @param render_every episodes
        max_frames: int
            maximum number of frames to train the agent in each episode
        """
        if episode <= 0 or render_every <= 0:
            raise ValueError("render_every and episode must be postive integers")
        if render_every > episode:
            raise ValueError("render_every must be less than episodes")

        self.reset()
        log_rewards = []
        cur_reward = 0
        for e in range(1,episode+1):
            observation, info = self.env.reset(seed=self.seed)

            done = False
            frames = []
            frame = 0
            while not done:
                frame += 1

                action = self.choose_action(observation)
                observation, reward, terminated, truncated, info = self.env.step(action)
                self.policy.rewards.append(reward)
                cur_reward += reward

                done = terminated or truncated
                if max_frames and (frame > max_frames):
                    done = True
                if self.render and not (e % render_every):
                    if self.rendergif:
                        frames.append(self.env.render())
                    else:
                        self.env.render()
            if self.rendergif and not (e % render_every):
                self.saveFramesToGif(
                    frames,
                    self.train_path,
                    f"Train{self.name}Episodes{e}Reward={cur_reward:.2f}.gif",
                )
            self.optimizer.zero_grad()
            loss = self.policy.loss(self.gamma)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            self.policy.reset()
            if not (e % save_every):
                model_path = os.path.join(
                    self.train_path,
                    f"train{self.name}Episode={e}Reward={cur_reward:.2f}.pth",
                )
                self.save(model_path)
                print((f"Model saved to {model_path}"))
            print(f"Train{self.name}\tEpisode={e}\tReward={cur_reward:.2f}")
            log_rewards.append(cur_reward)
        fig,ax = plt.subplots()
        ax.set_title(f"Reward results of training {self.name} for {episode} episodes")
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Reward")
        ax.plot(log_rewards)
        plt.savefig(os.path.join(self.train_path, "rewardlogs.png"))
        print(f"reward logs saved to {os.path.join(self.train_path,'rewardlogs.png')}")

    def save(self, path):
        """
        Save the agent

        Parameters:
        -----------
        path: str
            path to save the agent"""
        torch.save(self.policy.state_dict(), path)

    def test(self, model="", episode=12, max_frames=0,render_every=4):
        """
        Test the agent with the give path to the model otherwise, it tests the current model.

        Parameters:
        -----------
        model: str
            path to saved model, empty string tests the current model on the agent"
        episode: int
            number of episodes
        max_frames: int
            maximum number of frames to test the agent in each episode
        render_every: int
            render the environment every @param render_every episodes

        """
        if model:
            self.reset()
            self.policy = ActorCriticModule()
            self.policy.load_state_dict(torch.load(os.path.join(model)))

        observation, info = self.env.reset(seed=self.seed)
        cur_reward = 0
        for e in range(1,episode+1):
            observation, info = self.env.reset(seed=self.seed)
            done = False
            frames = []
            frame = 0
            while not done:
                frame += 1

                action = self.choose_action(observation)
                observation, reward, terminated, truncated, info = self.env.step(action)
                self.policy.rewards.append(reward)
                cur_reward += reward

                done = terminated or truncated
                if max_frames and (frame > max_frames):
                    done = True
                if self.render and not (e % render_every):
                    if self.rendergif:
                        frames.append(self.env.render())
                    else:
                        self.env.render()
            if self.rendergif and not (e % render_every):
                self.saveFramesToGif(
                    frames,
                    self.test_path,
                    f"Test{self.name}Episodes{e}Reward={cur_reward:.2f}.gif",
                )
            print(f"Test{self.name}\tEpisode={e}\tReward={cur_reward:.2f}")
