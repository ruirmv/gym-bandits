import gym
from gym import spaces, utils
from gym.utils import seeding
import numpy as np


class BanditsEnv(gym.Env):
    """
    Multi-Armed Bandit environment works as follows: at each time step the agent selects
    an action that corresponds to the selected bandit. No observation is made and a reward is received.
    The episode is infinite.
    
    There are no observations.
    
    Actions consist in selecting a bandit.
    
    The reward is a random sample taken from the bandit.
    
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, bandits=10):
        self._mean = 0
        self._standard_deviation = 1
        self._standard_deviation_reward = 1
        self._bandits = bandits
        self._expected_rewards = None
        self.action_space = spaces.Discrete(bandits)
        self.observation_space = spaces.Discrete(0)
        self.np_random = None
        
        if bandits < 1:
            raise ValueError("Number of bandits most be a positive integer.")
        
        self.seed()
        self.reset()
    
    def step(self, action):
        assert self.action_space.contains(action)
        
        reward = np.random.normal(loc=self._expected_rewards[action],
                                  scale=self._standard_deviation_reward)
        done = False
        obs = 0
        
        return obs, reward, done, {}
    
    def reset(self):
        self._expected_rewards = np.random.normal(loc=self._mean,
                                                  scale=self._standard_deviation,
                                                  size=self._bandits)
        
        return None
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def render(self, mode='human'):
        raise NotImplemented
    
    def close(self):
        raise NotImplemented
    
    def get_values(self):
        return self._expected_rewards.copy()
