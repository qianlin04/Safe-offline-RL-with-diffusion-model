import gym
import diffuser.environments
import numpy as np
import torch

env = gym.make("TwoStepMDP-v0")
dataset = {'actions': [], 'observations': [], 'rewards': [], 'terminals': [], 'timeouts': []}
for _ in range(1000):
    done = False
    obs = env.reset()
    while not done:
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        dataset['actions'].append(action)
        dataset['observations'].append(obs)
        dataset['rewards'].append(reward)
        dataset['terminals'].append(False)
        dataset['timeouts'].append(done)
        obs = next_obs

for k in dataset:
    dataset[k] = np.vstack(dataset[k])
dataset['rewards'] = dataset['rewards'].squeeze()
dataset['terminals'] = dataset['terminals'].squeeze()
dataset['timeouts'] = dataset['timeouts'].squeeze()

print(dataset['terminals'], dataset['timeouts'])
print(dataset['observations'])

torch.save(dataset, './two_step_mdp_experiment/dataset.pkl')
dataset = torch.load('./two_step_mdp_experiment/dataset.pkl')

print(dataset.keys())
print(dataset["observations"].shape, dataset["actions"].shape, dataset["rewards"].shape, dataset["terminals"].shape)