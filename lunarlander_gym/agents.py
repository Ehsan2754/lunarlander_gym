"""
lunarlander_gym Agents
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np



class BaseAgent(nn.Module):
    """Base Agent class 
    defines the interface to possible agents"""

    def __init__(self,env,name='Base Agent',seed=543):
        """
        Parameters:
        -----------
        env: gym.Env
        name: str
            name of the agent
        seed: int
            random seed    
        """
        super(BaseAgent, self).__init__()
        self.name = name
        self.env = env
        self.action_space = np.arange(env.action_space.start,env.action_space.n)
        self.observation_space = env.observation_space
        self.seed = seed
        torch.manual_seed(self.seed)


    def __str__(self) -> str:
        return f"""
        Name: {self.name}
        Action_space: {self.action_space}
        Environment: {self.env}
        """

    def reset(self):
        """
        Reset the agent
        """
        raise NotImplementedError

    def choose_action(self,observation):
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
    
    def save(self,path):
        """
        Save the agent
        
        Parameters:
        -----------
        path: str
            path to save the agent"""
    
    def test(self,model,gif=False):
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
    def __init__(self,env,name='Policy Gradient Agent',lr=1e-1,reward_decay=9.5e-1):
        """
        Parameters:
        -----------
        """
        super().__init__(env,name)
        self.action_space = np.arange(env.action_space.start,env.action_space.n)
        self.observation_space = env.observation_space
        self.lr = lr
        self.reward_decay = reward_decay

        self.memory = []



class ActorCriticAget(BaseAgent):
    """
    Actor Critic Reinforcement Agent
    """
    def __init__(self, env, name='Actor Critic Agent',gamma=9.9e-1, seed=543):
        super().__init__(env, name, seed)
        self.gamma = gamma
        self.affine = nn.Linear(8, 128)
        self.action_layer = nn.Linear(128, 4)
        self.value_layer = nn.Linear(128, 1)

        self.action_porbs = []
        self.state_values = []
        self.rewards = []

    def forward(self, state):

        out = torch.from_numpy(state).float()
        out = F.relu(self.affine(out))

        state_value = self.value_layer(out)
        action_probs = F.softmax(self.action_layer(out))
        action_distribution = Categorical(action_probs)

        action = action_distribution.sample()

        self.action_porbs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)

        return action.item()
    
    def loss(self):
        
        # calculating the discounted rewards
        rewards = []
        discounted_reward = 0
        for reward in self.rewards[::-1]:
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

    def reset(self):
        """
        Reset the agent
        """
        self.state_values = []
        self.rewards = []
        self.logprobs = []
        self.env.reset(seed=self.seed)