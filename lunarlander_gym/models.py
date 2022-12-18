"""
Neural Network Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class PolicyGradientModule(nn.Module):
    """
    Policy Gradient Module
    """
    def __init__(self):
        super(PolicyGradientModule, self).__init__()
        self.state_layer = nn.Linear(8,128)
        self.action_layer = nn.Linear(128,4)
    
    def forward(self, observation):
        out = F.relu(self.state_layer(observation))
        out = F.softmax(self.action_layer(out),dim=-1)
        return out


class ActorCriticModule(nn.Module):
    def __init__(self):
        super(ActorCriticModule, self).__init__()

        self.affine = nn.Linear(8, 128)
        self.action_layer = nn.Linear(128, 4)
        self.value_layer = nn.Linear(128, 1)

        self.action_porbs = []
        self.state_values = []
        self.rewards = []

    def forward(self, observation):
        observation = torch.from_numpy(observation).float()
        observation = F.relu(self.affine(observation))
        state_value = self.value_layer(observation)
        action_probs = F.softmax(self.action_layer(observation))
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        self.action_porbs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)

        return action.item()

    def loss(self, gamma):

        """calculating the discounted rewards"""
        rewards = []
        discounted_reward = 0
        for reward in self.rewards[::-1]:
            discounted_reward = reward + gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())

        loss = 0
        for action_prob, value, reward in zip(
            self.action_porbs, self.state_values, rewards
        ):
            advantage = reward - value.item()
            actor_loss = -action_prob * advantage
            critic_loss = F.smooth_l1_loss(value, reward)
            loss += actor_loss + critic_loss
        return loss

    def reset(self):
        del self.state_values[:]
        del self.rewards[:]
        del self.action_porbs[:]
