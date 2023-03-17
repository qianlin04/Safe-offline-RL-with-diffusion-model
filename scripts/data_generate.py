import gym
import sys

import diffuser.environments
import numpy as np
import torch
import diffuser.utils as utils

import numpy as np
import os
from stable_baselines3 import DQN, SAC
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
from stable_baselines3.common.callbacks import BaseCallback


class CustomCallback(BaseCallback):
    def __init__(self, infos_buffer, verbose=0, ):
        super(CustomCallback, self).__init__(verbose)
        self.accum_cost = 0
        self.accum_reward = 0
        self.infos_buffer = infos_buffer

    def _on_step(self) -> bool:
        self.infos_buffer.append(self.locals['infos'])
        self.accum_cost += self.locals['infos'][0]['cost']
        self.accum_reward += self.locals['rewards']
        if "TimeLimit.truncated" in self.locals['infos'][0] or self.locals['dones']:
            print(self.accum_reward, self.accum_cost)
            self.accum_cost = 0
            self.accum_reward = 0
        return True
    


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'mycliffwalking-v0'
    datatype: str = 'random'
    total_step: int = 1000000
    idx2onehot: int = 0
    model1 : str = None
    model2 : str = None
    mix_sample_p : str = None
    exploration_rate: float = 0.0
    #config: str = 'config.locomotion'

args = Parser().parse_args()
env = gym.make(args.dataset)
dataset = {'actions': [], 'observations': [], 'rewards': [], 'terminals': [], 'timeouts': []}
infos_buffer = []
collect_infos_callback = CustomCallback(infos_buffer)

if args.datatype == 'medium-replay':
    # model = DQN("MlpPolicy", env, verbose=0, buffer_size=args.total_step, train_freq=1, gradient_steps=1, target_update_interval=5,
    #             exploration_initial_eps=1.0, exploration_final_eps=0.1,
    #             exploration_fraction=1.0, learning_rate=0.0001, gamma=0.95, batch_size=512, learning_starts=args.total_step//20)
    # model = DQN("MlpPolicy", env, verbose=0, buffer_size=args.total_step, exploration_initial_eps=1.0, exploration_final_eps=0.1, learning_rate=0.001,
    #             train_freq=1, gradient_steps=1,
    #             exploration_fraction=0.8, batch_size=512, learning_starts=args.total_step//20, target_update_interval=10)
    model = SAC("MlpPolicy", env, verbose=1, learning_rate=1e-3, buffer_size=args.total_step, )
    model.learn(total_timesteps=args.total_step, log_interval=5, callback=collect_infos_callback, )
    model.save(f"./dataset/{args.dataset}.agent")
    dataset['actions'] = model.replay_buffer.actions.squeeze(1)
    dataset['observations'] = model.replay_buffer.observations.squeeze(1)
    dataset['rewards'] = model.replay_buffer.rewards.squeeze(-1)
    dataset['terminals'] = model.replay_buffer.dones.squeeze(-1)
    dataset['timeouts'] = model.replay_buffer.timeouts.squeeze(-1)
    #dataset['terminals'] = np.logical_and(dataset['terminals'], 1-dataset['timeouts'])
    if 'cost' in infos_buffer[0][0]:
        dataset['costs'] = np.array([info[0]['cost'] for info in infos_buffer])
        print(dataset['costs'].shape)
        assert len(dataset['costs'])==dataset['actions'].shape[0]

elif args.datatype == 'random' or args.datatype == 'medium' or args.datatype == 'mix':
    if args.datatype == 'medium' or args.datatype == 'mix': 
        model1_path = f"./dataset/{args.dataset}.agent" if args.model1 is None else args.model1
        model = DQN.load(model1_path)
        model.exploration_rate = args.exploration_rate
    
    if args.datatype == 'mix':
        mix_sample_p = [1, 1, 0] if args.mix_sample_p is None else [float(p) for p in args.mix_sample_p.split('-')]
    elif args.datatype == 'medium':
        mix_sample_p = [0, 1, 0]
    elif args.datatype == 'random':
        mix_sample_p = [1, 0, 0]
    if args.model2 is not None:
        model2 = DQN.load(args.model2)
        model2.exploration_rate = args.exploration_rate
        mix_sample_p[2] = 1
    mix_sample_p = np.array(mix_sample_p) / np.sum(mix_sample_p)
    
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
        timeouts = False
        obs = env.reset()
        sample_class = np.random.choice([0, 1, 2], p=mix_sample_p)
        t = 0
        while not done and not timeouts:
            t += 1
            total_step += 1
            
            if sample_class==0:
                action = env.action_space.sample()
            elif sample_class==1:
                action, _states = model.predict(obs, deterministic=False)
            else:
                action, _states = model2.predict(obs, deterministic=False)    
            
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
print("reward :         ", np.sum(dataset['rewards']), np.sum(dataset['timeouts']))
if args.idx2onehot:
    dataset['observations'] = np.eye(env.observation_space.n)[dataset['observations'].squeeze()]
    dataset['actions'] = np.eye(env.action_space.n)[dataset['actions'].squeeze()]

print(dataset['terminals'][:600], dataset['timeouts'][:600])
print(dataset['observations'][:50], dataset['actions'][:50])

torch.save(dataset, f'./dataset/{args.dataset}_{args.datatype}.pkl')
dataset = torch.load(f'./dataset/{args.dataset}_{args.datatype}.pkl')

print(dataset.keys())
print(dataset["observations"].shape, dataset["actions"].shape, dataset["rewards"].shape, dataset["terminals"].shape)