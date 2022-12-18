"""
lunarlander_gym Agents
"""
import torch
import torch.optim as optim
import gymnasium as gym
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
        name="Base Agent",
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
        return f"""
        Name: {self.name},
        Environment: {self.env}
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

    def save(self, path):
        """
        Save the agent

        Parameters:
        -----------
        path: str
            path to save the agent"""

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


"""        
Policy Gradient Agent
"""


class PolicyGradientAgent(BaseAgent):
    """
    Policy Gradient Reinforcement Agent
    """

    def __init__(self, env, name="Policy Gradient Agent", lr=1e-1, reward_decay=9.5e-1):
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
        name="Actor Critic Agent",
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

    def train(self, episode=10_000, save_every=0, render_every=0):
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
        """
        if episode <= 0 or render_every <= 0:
            raise ValueError("render_every and episode must be postive integers")
        if render_every > episode:
            raise ValueError("render_every must be less than episodes")
        
        
        self.reset()
        
        cur_reward = 0
        for e in range(episode):
            observation, info = self.env.reset(seed=self.seed)
        
            done = False
            frame = 0
            while not done:
                frame += 1
        
                action = self.choose_action(observation)
                observation, reward, terminated, truncated, info = self.env.step(action)
                self.policy.rewards.append(reward)
                cur_reward += reward

                done = terminated or truncated

                if self.render and render_every:
                    if not e % render_every:
                        if self.rendergif:
                            raise NotImplemented(f"render gif not implemented")
                        else:
                            self.env.render()
            self.optimizer.zero_grad()
            loss = self.policy.loss(self.gamma)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            self.policy.reset()
            if episode % save_every == 0:
                model_path = os.path.join(
                    self.train_path, f"train_episode{e}_reward{cur_reward}:.2f.pth"
                )
                self.save(model_path)
                logging.info(f"Model saved to {model_path}")
        print('Episode',e,frame)

    def save(self, path):
        """
        Save the agent

        Parameters:
        -----------
        path: str
            path to save the agent"""
        torch.save(self.policy.state_dict(), path)

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
