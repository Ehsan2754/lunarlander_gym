"""
lunarlander_gym Agents
"""
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import animation
import os
from collections import defaultdict
import json

import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import random
# from lunarlander_gym.models import *
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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        random.seed(seed)
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
        print("\t", filename[:-4])
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


class RandomAgent(BaseAgent):
    """
    Random Agent
    """

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
        if episode <= 10 or render_every <= 0:
            raise ValueError("render_every and episode must be postive integers >=10")
        if render_every > episode:
            raise ValueError("render_every must be less than episodes")
        observation, info = self.env.reset(seed=self.seed)
        log_rewards = []
        for e in range(1, episode + 1):
            cur_reward = 0
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
            log_rewards.append(cur_reward)
            print(f"Test{self.name}\tEpisode={e}\tReward={cur_reward:.2f}")

            if self.rendergif and not (e % render_every):
                self.saveFramesToGif(
                    frames,
                    self.test_path,
                    f"Test{self.name}Episodes{e}Reward={cur_reward:.2f}.gif",
                )
        fig, ax = plt.subplots()
        ax.set_title(f"Reward results of training {self.name} for {episode} episodes")
        ax.set_xlabel("Episodes")
        ax.set_ylabel(
            f"Reward(mean last 10 episodes={np.mean(np.array(log_rewards[-10:])):.2f})"
        )
        ax.plot(log_rewards)
        plt.savefig(os.path.join(self.train_path, "rewardlogs.png"))
        print(f"reward logs saved to {os.path.join(self.train_path,'rewardlogs.png')}")


class QLearningAget(BaseAgent):
    """
    Q-Learning Agent
    """

    def __init__(
        self,
        render=True,
        rendergif=False,
        path="./output",
        name="QLearningAget",
        lr=0.2e-1,
        gamma=9.9e-1,
        epsilon_decay=1e-2,
        seed=543,
    ):
        super().__init__(render, rendergif, path, name, seed)
        self.qStates = defaultdict(float)
        self.eps = 1.0
        self.lr = lr
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self._decayEpsilon = lambda eps1, eps2: eps1 if eps1 < eps2 else eps1 * 0.996

    def train(self, episode=10_000, save_every=500, render_every=1000, max_frames=0):
        """
        Learn the environment by agent
        Parameters:
        -----------
        epsilon: float
            epsilon value for epsilon-greedy exploration
        episode: int
            number of episodes
        save_every: bool |int
            saves the model weights every @param save_every episodes
        render_every: int
            render the environment every @param render_every episodes
        max_frames: int
            maximum number of frames to train the agent in each episode
        """
        if episode <= 10 or render_every <= 0:
            raise ValueError("render_every and episode must be postive integers >=10")
        if render_every > episode:
            raise ValueError("render_every must be less than episodes")

        self.reset()
        log_rewards = []
        for e in range(1, episode + 1):
            observation, info = self.env.reset(seed=self.seed)
            observation = self._discretState(observation)

            cur_reward = 0
            done = False
            frames = []
            frame = 0

            while not done:
                frame += 1

                action = self.choose_action(observation)
                qState = str([observation, action])

                next_observation, reward, terminated, truncated, info = self.env.step(
                    action
                )

                next_observation = self._discretState(next_observation)

                self.qStates[qState] += self.lr * (
                    reward
                    + self.gamma * self._greedy(next_observation)
                    - self.qStates[qState]
                )

                observation = next_observation
                done = terminated or truncated

                if max_frames and (frame > max_frames):
                    done = True
                cur_reward += reward

                if self.render and not (e % render_every):
                    if self.rendergif:
                        frames.append(self.env.render())
                    else:
                        self.env.render()

            self.qStates[qState] += self.lr * (reward - self.qStates[qState])
            self.eps = self._decayEpsilon(self.eps, self.epsilon_decay)
            log_rewards.append(cur_reward)

            if self.rendergif and not (e % render_every):
                self.saveFramesToGif(
                    frames,
                    self.train_path,
                    f"Train{self.name}Episodes{e}Reward={cur_reward:.2f}.gif",
                )
            if not (e % save_every):
                model_path = os.path.join(
                    self.train_path,
                    f"train{self.name}Episode={e}Reward={cur_reward:.2f}.pth",
                )
                self.save(model_path)
                print((f"Model saved to {model_path}"))
            print(f"Train{self.name}\tEpisode={e}\tReward={cur_reward:.2f}")
            log_rewards.append(cur_reward)
        fig, ax = plt.subplots()
        ax.set_title(f"Reward results of training {self.name} for {episode} episodes")
        ax.set_xlabel("Episodes")
        ax.set_ylabel(
            f"Reward(mean last 10 episodes={np.mean(np.array(log_rewards[-10:])):.2f})"
        )
        ax.plot(log_rewards)
        plt.savefig(os.path.join(self.train_path, "rewardlogs.png"))
        print(f"reward logs saved to {os.path.join(self.train_path,'rewardlogs.png')}")

    def test(self, model="", episode=12, render_every=4, max_frames=0):
        """
        Learn the environment by agent
        Parameters:
        -----------
        model: str
            path to the saved model
        epsilon: float
            epsilon value for epsilon-greedy exploration
        episode: int
            number of episodes
        render_every: int
            render the environment every @param render_every episodes
        max_frames: int
            maximum number of frames to train the agent in each episode
        """
        if episode <= 10 or render_every <= 0:
            raise ValueError("render_every and episode must be postive integers >=10")
        if render_every > episode:
            raise ValueError("render_every must be less than episodes")
        if model:
            self.reset()
            with open(model, "r") as f:
                self.qStates, self.eps = json.loads(f.read())
                self.qStates = defaultdict(str, self.qStates)
                f.close()
        log_rewards = []
        for e in range(1, episode + 1):
            observation, info = self.env.reset(seed=self.seed)
            observation = self._discretState(observation)

            cur_reward = 0
            done = False
            frames = []
            frame = 0

            while not done:
                frame += 1

                action = self.choose_action(observation)
                qState = str([observation, action])

                next_observation, reward, terminated, truncated, info = self.env.step(
                    action
                )

                next_observation = self._discretState(next_observation)

                # self.qStates[qState] += self.lr * (reward+self.gamma*self._greedy(next_observation)-self.qStates[qState])

                observation = next_observation
                done = terminated or truncated

                if max_frames and (frame > max_frames):
                    done = True
                cur_reward += reward

                if self.render and not (e % render_every):
                    if self.rendergif:
                        frames.append(self.env.render())
                    else:
                        self.env.render()

            # self.qStates[qState] += self.lr * (reward - self.qStates[qState])
            # self.eps = self._decayEpsilon(self.eps,self.epsilon_decay)
            log_rewards.append(cur_reward)

            print(f"Test{self.name}\tEpisode={e}\tReward={cur_reward:.2f}")
            if self.rendergif and not (e % render_every):
                self.saveFramesToGif(
                    frames,
                    self.test_path,
                    f"Test{self.name}Episodes{e}Reward={cur_reward:.2f}.gif",
                )
            log_rewards.append(cur_reward)
        fig, ax = plt.subplots()
        ax.set_title(f"Reward results of testing {self.name} for {episode} episodes")
        ax.set_xlabel("Episodes")
        ax.set_ylabel(
            f"Reward(mean last 10 episodes={np.mean(np.array(log_rewards[-10:])):.2f})"
        )
        ax.plot(log_rewards)
        plt.savefig(os.path.join(self.train_path, "rewardlogs.png"))
        print(f"reward logs saved to {os.path.join(self.test_path,'rewardlogs.png')}")

    def _discretState(self, state):
        """
        Discretize the state of the environment
        Parameters:
        -----------
        state: ndarray
            state of the environment"""
        return (
            min(2, max(-2, int((state[0]) / 0.05))),
            min(2, max(-2, int((state[1]) / 0.1))),
            min(2, max(-2, int((state[2]) / 0.1))),
            min(2, max(-2, int((state[3]) / 0.1))),
            min(2, max(-2, int((state[4]) / 0.1))),
            min(2, max(-2, int((state[5]) / 0.1))),
            int(state[6]),
            int(state[7]),
        )

    def _greedy(self, observation):
        """
        Greedy policy for finding maximum reward in our Q table
        Parameters:
        -----------
        observation: ndarray
            next state of the environment
        """
        return np.max(
            [
                self.qStates[str([observation, action])]
                for action in np.arange(
                    self.env.action_space.start, self.env.action_space.n
                )
            ]
        )

    def choose_action(self, observation):
        """
        Choose an action based on epsilon-greedy policy
        Parameters:
        -----------
        observation: ndarray
            next state of the environment
        eps: float
            epsilon value
        """
        prob = np.random.random()
        if prob < self.eps:
            return random.choice(
                np.arange(self.env.action_space.start, self.env.action_space.n)
            )
        else:
            return np.argmax(
                [
                    self.qStates[str([observation, action])]
                    for action in np.arange(
                        self.env.action_space.start, self.env.action_space.n
                    )
                ]
            )

    def save(self, path):
        """
        Parameters:
        -----------
        path: str
            path to save the model"""
        with open(path, "w+") as f:
            f.write(json.dumps([self.qStates, self.eps]))
            f.close()

    def reset(self):
        self.qStates = defaultdict(float)
        self.eps = 1.0


class ActorCriticAgent(BaseAgent):
    """
    Actor Critic Reinforcement Agent
    (VPG + baseline)
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
        if episode <= 10 or render_every <= 0:
            raise ValueError("render_every and episode must be postive integers >=10")
        if render_every > episode:
            raise ValueError("render_every must be less than episodes")

        self.reset()
        log_rewards = []
        for e in range(1, episode + 1):
            cur_reward = 0
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
        fig, ax = plt.subplots()
        ax.set_title(f"Reward results of training {self.name} for {episode} episodes")
        ax.set_xlabel("Episodes")
        ax.set_ylabel(
            f"Reward(mean last 10 episodes={np.mean(np.array(log_rewards[-10:])):.2f})"
        )
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

    def test(self, model="", episode=12, max_frames=0, render_every=4):
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
        log_rewards = []
        for e in range(1, episode + 1):
            cur_reward = 0
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
            log_rewards.append(cur_reward)
            print(f"Test{self.name}\tEpisode={e}\tReward={cur_reward:.2f}")
        fig, ax = plt.subplots()
        ax.set_title(f"Reward results of testing {self.name} for {episode} episodes")
        ax.set_xlabel("Episodes")
        ax.set_ylabel(
            f"Reward(mean last 10 episodes={np.mean(np.array(log_rewards[-10:])):.2f})"
        )
        ax.plot(log_rewards)
        plt.savefig(os.path.join(self.train_path, "rewardlogs.png"))
        print(f"reward logs saved to {os.path.join(self.train_path,'rewardlogs.png')}")


class VanillaPolicyGradientAgent(BaseAgent):
    """
    Automatic Differentiation Agent

    PS. Since Actro Critic Agent uses AD for training the models, I would introduce ActorCriticAgent as my AD agent. 
    """

    def __init__(
        self,
        render=True,
        rendergif=False,
        path="./output",
        name="VanillaPolicyGradientAgent",
        lr=0.1e-1,
        gamma=9.8e-1,
        batch=32,
        seed=543,
    ):
        super().__init__(render, rendergif, path, name, seed)
        self.env.reset(seed=self.seed)
        self.lr = lr
        self.gamma = gamma
        self.policy = PolicyGradientModule()
        self.optimizer = optim.Adam(
            self.policy.parameters(), lr=self.lr,
        )
        self.batch = batch

    def reset(self):
        """
        Reset the agent
        """
        self.env.reset(seed=self.seed)
        self.policy = PolicyGradientModule()
        self.optimizer = optim.Adam(
            self.policy.parameters(), lr=self.lr
        )

    def choose_action(self, observation):
        """
        Step the agent
        Parameters:
        -----------
        observation: np.array of shape (n_samples,...)
            state vector of the environment
        """
        probability = self.policy(observation)
        c = Categorical(probability) 
        action = c.sample()
        return action.data.numpy().astype('int32')

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
        if episode <= 10 or render_every <= 0:
            raise ValueError("render_every and episode must be postive integers >=10")
        if render_every > episode:
            raise ValueError("render_every must be less than episodes")

        self.reset()
        steps = 0
        log_states = []
        log_actions = []
        log_rewards = []
        for e in range(1, episode + 1):
            cur_reward = 0
            observation, info = self.env.reset(seed=self.seed)
            observation = Variable(torch.from_numpy(observation).float())
            done = False
            frames = []
            frame = 0
            while not done:
                frame += 1

                action = self.choose_action(observation)
                next_observation, reward, terminated, truncated, info = self.env.step(action)

                done = terminated or truncated
                if max_frames and (frame > max_frames):
                    done = True
                reward = 0 if done else reward
                cur_reward += reward
                if self.render and not (e % render_every):
                    if self.rendergif:
                        frames.append(self.env.render())
                    else:
                        self.env.render()
                log_states.append(observation)
                log_actions.append(float(action))
                log_rewards.append(reward)
                observation = Variable(torch.from_numpy(next_observation).float())
                steps+=1

                if not (episode%self.batch):
                    r = 0
                    for i in reversed(range(steps)):
                        if not log_rewards[i]:
                            r = r*self.gamma + log_rewards[i]
                    norm_rewards = np.array(log_rewards)
                    norm_rewards = (norm_rewards - np.mean(norm_rewards))/np.std(norm_rewards)

                    self.optimizer.zero_grad()

                    for i in range(steps):
                        observation = log_states[i]
                        action = Variable(torch.FloatTensor(log_actions[i]))
                        reward = norm_rewards[i]
                        probabilities = self.policy(observation)
                        c = Categorical(probabilities)
                        loss = -c.log_prob(action)*reward
                        loss.backward()
                    self.optimizer.step()
                    log_actions.clear()
                    log_rewards.clear()
                    log_states.clear()
                    
            if self.rendergif and not (e % render_every):
                self.saveFramesToGif(
                    frames,
                    self.train_path,
                    f"Train{self.name}Episodes{e}Reward={cur_reward:.2f}.gif",
                )
            if not (e % save_every):
                model_path = os.path.join(
                    self.train_path,
                    f"train{self.name}Episode={e}Reward={cur_reward:.2f}.pth",
                )
                self.save(model_path)
                print((f"Model saved to {model_path}"))
            print(f"Train{self.name}\tEpisode={e}\tReward={cur_reward:.2f}")
            log_rewards.append(cur_reward)
        fig, ax = plt.subplots()
        ax.set_title(f"Reward results of training {self.name} for {episode} episodes")
        ax.set_xlabel("Episodes")
        ax.set_ylabel(
            f"Reward(mean last 10 episodes={np.mean(np.array(log_rewards[-10:])):.2f})"
        )
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

    def test(self, model="", episode=12, max_frames=0, render_every=4):
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
            self.policy = PolicyGradientModule()
            self.policy.load_state_dict(torch.load(os.path.join(model)))

        
        self.reset()
        steps = 0
        log_states = []
        log_actions = []
        log_rewards = []
        for e in range(1, episode + 1):
            cur_reward = 0
            observation, info = self.env.reset(seed=self.seed)
            observation = Variable(torch.from_numpy(observation).float())
            done = False
            frames = []
            frame = 0
            while not done:
                frame += 1

                action = self.choose_action(observation)
                next_observation, reward, terminated, truncated, info = self.env.step(action)

                done = terminated or truncated
                if max_frames and (frame > max_frames):
                    done = True
                reward = 0 if done else reward
                cur_reward += reward
                if self.render and not (e % render_every):
                    if self.rendergif:
                        frames.append(self.env.render())
                    else:
                        self.env.render()
                log_states.append(observation)
                log_actions.append(float(action))
                log_rewards.append(reward)
                observation = Variable(torch.from_numpy(next_observation).float())
                steps+=1

                if not (episode%self.batch):
                    r = 0
                    for i in reversed(range(steps)):
                        if not log_rewards[i]:
                            r = r*self.gamma + log_rewards[i]
                    norm_rewards = np.array(log_rewards)
                    norm_rewards = (norm_rewards - np.mean(norm_rewards))/np.std(norm_rewards)

                    self.optimizer.zero_grad()

                    for i in range(steps):
                        observation = log_states[i]
                        action = Variable(torch.FloatTensor(log_actions[i]))
                        reward = norm_rewards[i]
                        probabilities = self.policy(observation)
                        c = Categorical(probabilities)
                        # loss = -c.log_prob(action)*reward
                        # loss.backward()
                    # self.optimizer.step()
                    log_actions.clear()
                    log_rewards.clear()
                    log_states.clear()
                    
            if self.rendergif and not (e % render_every):
                self.saveFramesToGif(
                    frames,
                    self.test_path,
                    f"Test{self.name}Episodes{e}Reward={cur_reward:.2f}.gif",
                )
            print(f"Test{self.name}\tEpisode={e}\tReward={cur_reward:.2f}")
            log_rewards.append(cur_reward)
        fig, ax = plt.subplots()
        ax.set_title(f"Reward results of testing {self.name} for {episode} episodes")
        ax.set_xlabel("Episodes")
        ax.set_ylabel(
            f"Reward(mean last 10 episodes={np.mean(np.array(log_rewards[-10:])):.2f})"
        )
        ax.plot(log_rewards)
        plt.savefig(os.path.join(self.test_path, "rewardlogs.png"))
        print(f"reward logs saved to {os.path.join(self.test_path,'rewardlogs.png')}")