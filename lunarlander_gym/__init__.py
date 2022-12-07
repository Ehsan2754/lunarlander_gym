"""Top-level package for lunarlander_gym."""

__author__ = """Ehsan Shaghaei"""
__email__ = 'ehsan2754@gmail.com'
__status__ = 'Development'
__version__ = '0.1.0'

import sys
import os


class BaseAgent():
    """Base Agent class 
    defines the interface to possible agents"""

    def __init__(self,env):
        self.name = 'Base Agent'
        self.action_space = None
        self.observation_space = None
        self.env = env
        self.env_core = None
