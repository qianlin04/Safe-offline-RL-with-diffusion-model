import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from typing import Callable, List, Dict, Tuple
import torch
from os import path
from typing import Union
from gym import Env
from diffuser.environments.wrappers import SafeEnv, OfflineEnv

Array = Union[torch.Tensor, np.ndarray]

class OCPMEnv(gym.Env):

    def __init__(
            self, 
            dataset_name:str=None, batch_size=100, max_test_sample=20000):
        self.dataset_name = dataset_name
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(33,))
        self.action_space = spaces.Discrete(20)
        self._max_episode_steps = int(200)

        self.train_dataset = torch.load(f'./dataset/ocpm_train.pkl')
        self.test_dataset = torch.load(f'./dataset/ocpm_test.pkl')

        for d in self.test_dataset:
            self.test_dataset[d] = self.test_dataset[d][:max_test_sample]

        self.data_id = 0
        self.max_test_sample = max_test_sample
        self.batch_size = batch_size


    def step(self, action:np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        print(np.argmax(action, 1), '\n', np.max(action, 1))

        self.test_dataset['actions'][self.data_id:self.data_id+self.batch_size] = np.argmax(action, 1)
        self.data_id += self.batch_size
        done = (self.data_id >= len(self.test_dataset['observations']))        
        if done:
            next_obs = self.test_dataset['observations'][0:self.batch_size]
            torch.save(self.test_dataset, f"./dataset/ocpm_test_results_{self.max_test_sample}.pkl")
        else:
            next_obs = self.test_dataset['observations'][self.data_id:self.data_id+self.batch_size]
        reward = 0
        return next_obs, reward, done, {'cost': 0}

    def get_budget(self):
        if self.data_id == len(self.test_dataset['observations']):
            return 0
        else:
            return self.test_dataset['residual_constraint_v1'][self.data_id:self.data_id+self.batch_size]

    def reset(self) -> np.ndarray:
        self.data_id = 0
        return self.test_dataset['observations'][0:self.batch_size]

    def get_dataset(self, dataset=None):
        return self.train_dataset