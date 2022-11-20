import gym
import diffuser.environments
import numpy as np
import torch
import diffuser.utils as utils

import numpy as np
import os
from stable_baselines3 import DQN
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'cliffwalking_random-v0'
    datatype: str = 'random'
    total_step: int = 1000000
    idx2onehot: int = 1
    #config: str = 'config.locomotion'

args = Parser().parse_args()
env = gym.make(args.dataset, onehot=False)
dataset = {'actions': [], 'observations': [], 'rewards': [], 'terminals': [], 'timeouts': []}

if args.datatype == 'medium-replay':
    model = DQN("MlpPolicy", env, verbose=1, buffer_size=args.total_step, exploration_fraction=0.8, learning_rate=0.01, batch_size=512)
    model.learn(total_timesteps=args.total_step, log_interval=4)
    model.save(f"./dataset/{args.dataset}.agent")
    dataset['actions'] = model.replay_buffer.actions.squeeze(-1)
    dataset['observations'] = model.replay_buffer.observations.squeeze(-1)
    dataset['rewards'] = model.replay_buffer.rewards.squeeze(-1)
    dataset['terminals'] = model.replay_buffer.dones.squeeze(-1)
    dataset['timeouts'] = model.replay_buffer.timeouts.squeeze(-1)
    dataset['terminals'] = np.logical_and(dataset['terminals'], 1-dataset['timeouts'])

elif args.datatype == 'random' or args.datatype == 'medium' or args.datatype == 'mix':
    if args.datatype == 'medium' or args.datatype == 'mix': model = DQN.load(f"./dataset/{args.dataset}.agent")
    model.exploration_rate = 0.3 if args.datatype == 'mix' else 0.0 
    # x = np.zeros(env.shape, np.uint)
    # dire = ['U', 'R', 'D', 'L']
    # model.predict(3*12+0)
    # model.predict(2*12+0)
    # model.predict(2*12+1)
    # model.predict(2*12+2)
    # model.predict(1*12+0)
    # model.predict(1*12+1)
    # print("\n")

    
    # model.predict(2*12+10)
    # model.predict(2*12+11)
    
    # print("\n")
    # #assert 0
    # for i in range(48):
    #     action, _states = model.predict(i, deterministic=True)
    #     a,b = np.unravel_index(i, env.shape)
    #     x[a, b]=int(action)
    # for i in range(4):
    #     p = "".join([dire[int(j)] for j in x[i]])
    #     print(p)
    # assert 0
    total_step = 0
    while total_step<args.total_step:
        done = False
        obs = env.reset()
        t = 0
        while not done:
            t += 1
            total_step += 1
            if args.datatype == 'random':
                action = env.action_space.sample()
            else:
                action, _states = model.predict(obs, deterministic=False)
            action = int(action)
            next_obs, reward, done, info = env.step(action)
            dataset['actions'].append(action)
            dataset['observations'].append(obs)
            dataset['rewards'].append(reward)
            #print(obs, action, next_obs, reward, done)
            timeouts = info.get("TimeLimit.truncated", False) 
            dataset['terminals'].append(done and not timeouts)
            dataset['timeouts'].append(timeouts)
            obs = next_obs

    for k in dataset:
        dataset[k] = np.vstack(dataset[k])
    dataset['rewards'] = dataset['rewards'].squeeze()
    dataset['terminals'] = dataset['terminals'].squeeze()
    dataset['timeouts'] = dataset['timeouts'].squeeze()

if args.idx2onehot:
    dataset['observations'] = np.eye(env.nS)[dataset['observations'].squeeze()]
    dataset['actions'] = np.eye(env.nA)[dataset['actions'].squeeze()]

print(dataset['terminals'][:200], dataset['timeouts'][:200])
print(dataset['observations'][:5], dataset['actions'][:5])

torch.save(dataset, f'./dataset/{args.dataset}_{args.datatype}.pkl')
dataset = torch.load(f'./dataset/{args.dataset}_{args.datatype}.pkl')

print(dataset.keys())
print(dataset["observations"].shape, dataset["actions"].shape, dataset["rewards"].shape, dataset["terminals"].shape)