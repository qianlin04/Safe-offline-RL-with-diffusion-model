import numpy as np
import pandas as pd
# 定义非线性系统环境，按照GYM的格式
import gym
from gym import spaces, logger
from gym.utils import seeding
import torch

class TwoStepMDP(gym.Env):
    def __init__(self, onehot=True):
        self.state = np.array([1, 0, 0, 0])
        self.rewards = np.array([0, 0, 2, -3])
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4, ), dtype=np.float64)
        self._max_episode_steps = 8
    
    def get_dataset(self, dataset='./two_step_mdp_experiment/dataset.pkl'):
        return torch.load(dataset)

    def onehot2n(self, onehot):
        return np.sum(onehot*np.array([0, 1, 2, 3]))

    def reset(self, init_state=None):
        self.t = 0
        self.state = np.array([0, 0, 0, 0])
        self.state[np.random.randint(0, 4)] = 1
        if init_state is not None:
            self.state = np.array(init_state)
        return self.state
    
    def step(self, action):
        n = int(self.onehot2n(self.state))
        if n==0:
            if action>0.5: 
                self.state = np.array([0, 1, 0, 0])
            else:
                if np.random.uniform(0, 1)>0.5:
                    self.state = np.array([0, 0, 1, 0])
                else:
                    self.state = np.array([0, 0, 0, 1])
        else:
            self.state = np.array([1, 0, 0, 0])

        reward = self.rewards[n]
        self.t += 1
        done = (self.t>=self._max_episode_steps)
        info = {}
        return self.state, reward, done, info
    
    def render(self):
        pass

    def state_vector(self):
        return self.state

    def get_normalized_score(self, total_reward):
        return 0

