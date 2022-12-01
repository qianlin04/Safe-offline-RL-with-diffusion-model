import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class RouletteEnv(gym.Env):
    """Simple roulette environment

    The roulette wheel has 37 spots. If the bet is 0 and a 0 comes up,
    you win a reward of 35. If the parity of your bet matches the parity
    of the spin, you win 1. Otherwise you receive a reward of -1.

    The long run reward for playing 0 should be -1/37 for any state

    The last action (38) stops the rollout for a return of 0 (walking away)
    """
    def __init__(self, spots=5, onehot=True, datatype=''):
        self.datatype = datatype
        self.onehot = onehot
        self.n = spots + 1
        self.action_space = spaces.Discrete(self.n)
        self.observation_space = spaces.Discrete(3)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if type(action) is np.ndarray:
            action = np.argmax(action)
        
        assert self.action_space.contains(action)

        # N.B. np.random.randint draws from [A, B) while random.randint draws from [A,B]
        val = self.np_random.randint(0, self.n - 1)
        if val == action == 0:
            reward = self.n - 2.0
        elif val != 0 and action != 0 and val % 2 == action % 2:
            reward = 1.0
        else:
            reward = -1.0
        state = 2 if val==0 else val % 2
        if self.onehot: state = np.eye(self.observation_space.n)[state] 

        if action == self.n - 1:
            # observation, reward, done, info
            return state, 0, True, {}

        return state, reward, False, {}

    def reset(self):
        val = self.np_random.randint(0, self.n - 1)
        state = 2 if val==0 else val % 2
        if self.onehot: return np.eye(self.observation_space.n)[state]
        return 0

    def get_dataset(self, dataset=None):
        import torch
        if dataset is None:
            dataset = f'./dataset/myroulette-v0_{self.datatype}.pkl' 
        return torch.load(dataset)