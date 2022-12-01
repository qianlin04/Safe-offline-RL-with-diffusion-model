import socket

from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ## value kwargs
    ('discount', 'd'),
]

logbase = 'logs'

base = {
    'diffusion': {
        ## model
        'model': 'models.TemporalConvNet', #'models.TemporalUnet_twostepmdp',
        'diffusion': 'models.PolicyDynamicDiffusion',
        'horizon': 8,
        'n_diffusion_steps': 20,
        'action_weight': 10,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 2, 4),
        'attention': False,
        'renderer': None,

        ## dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'DebugNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 200,

        ## serialization
        'logbase': logbase,
        'prefix': 'decouple_diffusion_casual/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 1000,
        'loss_type': 'l2',
        'n_train_steps': 1e6,
        'batch_size': 128,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 10000,
        'sample_freq': 10000,
        'n_saves': 5,
        'save_parallel': False,
        'bucket': None,
        'device': 'cuda',
        'seed': None,
        'n_reference': 8,
        'predict_reward_done': False,
    },

    'values': {
        'model': 'models.TemporalVauleNet',
        'diffusion': 'models.DecoupleValueDiffusion',
        'horizon': 8,
        'n_diffusion_steps': 20,
        'dim_mults': (1, 2, 4),
        'renderer': None,

        ## value-specific kwargs
        'discount': 0.99,
        'termination_penalty': 0,
        'normed': False,

        ## dataset
        'loader': 'datasets.ValueDataset',
        'normalizer': 'DebugNormalizer',
        'preprocess_fns': [],
        'use_padding': True,
        'max_path_length': 200,

        ## serialization
        'logbase': logbase,
        'prefix': 'values/decouple',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 1000,
        'loss_type': 'value_l2',
        'n_train_steps': 1e6,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 0,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda',
        'seed': None,
    },

    'cost_values': {
        'model': 'models.ValueFunction',
        'diffusion': 'models.ValueDiffusion',
        'horizon': 8,
        'n_diffusion_steps': 20,
        'dim_mults': (1, 2, 4),
        'renderer': None,

        ## value-specific kwargs
        'discount': 0.99,
        'termination_penalty': 0,
        'normed': False,

        ## dataset
        'loader': 'datasets.ValueDataset',
        'normalizer': 'DebugNormalizer',
        'preprocess_fns': [],
        'use_padding': True,
        'max_path_length': 200,

        ## serialization
        'logbase': logbase,
        'prefix': 'vel_cost_values/decouple',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 1000,
        'loss_type': 'value_l2',
        'n_train_steps': 50000,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 0,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda',
        'seed': None,
    },

    'plan': {
        'guide': 'sampling.ValueGuide',
        'policy': 'sampling.GuidedPolicy',
        'max_episode_length': 200,
        'batch_size': 1,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': None,

        ## sample_kwargs
        'n_guide_steps': 2,
        'scale': 0.1,
        't_stopgrad': 2,
        'scale_grad_by_std': True,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/',
        'exp_name': watch(args_to_watch),
        'vis_freq': 100,
        'max_render': 8,

        ## diffusion model
        'horizon': 8,
        'n_diffusion_steps': 20,

        ## value function
        'discount': 0.99,

        ## loading
        'diffusion_loadpath': 'f:decouple_diffusion_casual/defaults_H{horizon}_T{n_diffusion_steps}',
        'value_loadpath': 'f:values/decouple_H{horizon}_T{n_diffusion_steps}_d{discount}',
        'cost_value_loadpath': 'f:action_cost_values/decouple_H{horizon}_T{n_diffusion_steps}_d{discount}',

        'cost_grad_weight': 1.0,

        'diffusion_epoch': 'latest',
        'value_epoch': 'latest',
        'cost_value_epoch': 'latest',

        'verbose': False,
        'suffix': '0',

        'state_grad_mask': False,
    },

    'policylearning': {
        ## model
        'model': 'models.TemporalConvNet',
        'diffusion': 'models.PolicyDynamicDiffusion',
        'agent_algo': 'COMBOPolicy',
        'agent': 'f:agent.{agent_algo}',
        'agent_params': {
            'conservative_weight': 5.0,
            'n_action_samples': 10,
            'real_ratio': 0.05,
        },
        'horizon': 8,
        'n_diffusion_steps': 20,
        'action_weight': 10,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 2, 4),
        'attention': False,
        'renderer': None,

        ## dataset
        'replaybuffer': 'datasets.ReplayBuffer',
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'DebugNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 200,

        ## serialization
        'logbase': logbase,
        'prefix': 'f:decouple_policylearning_{agent_algo}_{seed}/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 1000,
        'loss_type': 'l2',
        'n_train_steps': 2e6,
        'batch_size': 128,
        'learning_rate': 2e-4,
        'actor_learning_rate': 2e-4,
        'critic_learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 10000,
        'sample_freq': 10000,
        'n_saves': 5,
        'save_parallel': False,
        'bucket': None,
        'device': 'cuda',
        'seed': 0,
        'n_reference': 8,

        'rollout_batch_size': 50000,
        'rollout_length': 5, 
        'model_retain_epochs': 5,
        'rollout_freq': 1000,
        'real_ratio': 0.05,
        'agent_batch_size': 256,
        'eval_episodes': 10,
        'load_epoch': None,
        'train_policy_freq': 1, 
        'warmup_step': 4e5,

        'use_wandb': True,
        'group': 'default',
        'algo': 'f:{agent_algo}_diffusion', 
    },
}

mycliffwalking_mix_v0 = {
    'diffusion': {
        'prefix': 'decouple_reward_model/defaults',
        'predict_reward_done': True,
    },
    'plan': {
        #'diffusion_loadpath': 'f:decouple_policylearning/defaults_H{horizon}_T{n_diffusion_steps}',
    }, 
    'policylearning': {
        'load_epoch': 790000,
        'rollout_batch_size': 1200,
        'rollout_length': 8,
        'seed': 0,
        'real_ratio': 0.05,
        'agent_params': {
            'conservative_weight': 5.0,
            'n_action_samples': 10,
            'real_ratio': 0.05,
        },
        'use_wandb':False, 
    }
}

myroulette_random_v0 = {
    'diffusion': {
        'horizon': 4,
    },
    'values': {
        'horizon': 4,
    },
    'plan': {
        'horizon': 4,
    },
}

myFrozenLake_medium_replay_v0 = {
    'diffusion': {
        'horizon': 4,
        
    },
    'values': {
        'horizon': 4,
        
    },
    'plan': {
        'horizon': 4,
    },
}

TwoStepMDP_v0 ={
    'diffusion': {
        'horizon': 4,
        
    },
    'values': {
        'horizon': 4,
        
    },
    'plan': {
        'horizon': 4,
    },
}