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
        'model': 'models.TemporalConvNet',
        'diffusion': 'models.PolicyDynamicDiffusion',
        'horizon': 32,
        'n_diffusion_steps': 20,
        'action_weight': 10,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 2, 4, 2, 1),
        'attention': False,
        'renderer': 'utils.MuJoCoRenderer',

        ## dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'GaussianNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 1000,

        ## serialization
        'logbase': logbase,
        'prefix': 'decouple_diffusion_casual/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 1e6,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 20000,
        'sample_freq': 20000,
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
        'horizon': 32,
        'n_diffusion_steps': 20,
        'dim_mults': (1, 2, 4, 2, 1),
        'renderer': 'utils.MuJoCoRenderer',

        ## value-specific kwargs
        'discount': 0.997,
        'termination_penalty': -100,
        'normed': False,

        ## dataset
        'loader': 'datasets.ValueDataset',
        'normalizer': 'GaussianNormalizer',
        'preprocess_fns': [],
        'use_padding': True,
        'max_path_length': 1000,

        ## serialization
        'logbase': logbase,
        'prefix': 'values/decouple',
        'exp_name': watch(args_to_watch),

        ## trainingd
        'n_steps_per_epoch': 10000,
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
        'model': 'models.TemporalVauleNet',
        'diffusion': 'models.DecoupleValueDiffusion',
        'horizon': 32,
        'n_diffusion_steps': 20,
        'dim_mults': (1, 2, 4, 2, 1),
        'renderer': 'utils.MuJoCoRenderer',

        ## value-specific kwargs
        'discount': 0.997,
        'termination_penalty': 1000,
        'normed': False,

        ## dataset
        'loader': 'datasets.ValueDataset',
        'normalizer': 'GaussianNormalizer',
        'preprocess_fns': [],
        'use_padding': True,
        'max_path_length': 1000,

        ## serialization
        'logbase': logbase,
        'prefix': 'vel_cost_values/decouple',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'value_l2',
        'n_train_steps': 400e3,
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
        'max_episode_length': 1000,
        'batch_size': 64,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': None,

        ## sample_kwargs
        'n_guide_steps': 2,
        'scale': 0.1,
        't_stopgrad': 2,
        'scale_grad_by_std': True,

        ## serializationqs
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/',
        'exp_name': watch(args_to_watch),
        'vis_freq': 100,
        'max_render': 8,

        ## diffusion model
        'horizon': 32,
        'n_diffusion_steps': 20,

        ## value function
        'discount': 0.997,

        ## loading
        'diffusion_loadpath': 'f:decouple_diffusion_casual_sync/defaults_H{horizon}_T{n_diffusion_steps}',
        'value_loadpath': 'f:values/defaults_H{horizon}_T{n_diffusion_steps}_d{discount}',
        'cost_value_loadpath': 'f:vel_cost_values/decouple_H{horizon}_T{n_diffusion_steps}_d{discount}',

        'cost_grad_weight': 10.0,

        'diffusion_epoch': 'latest',
        'value_epoch': 'latest',
        'cost_value_epoch': 'latest',

        'verbose': False,
        'suffix': '0',

        'state_grad_mask': False,
        'test_cost_with_discount': True,
    },

    'policylearning': {
        ## model
        'model': 'models.TemporalConvNet',
        'diffusion': 'models.PolicyDynamicDiffusion',
        'agent_algo': 'COMBOPolicy',
        'agent': 'f:agent.{agent_algo}',
        'agent_params': {
            'COMBOPolicy': {
                'conservative_weight': 1.0,
                'n_action_samples': 10,
            },
            'CQLPolicy': {
                'conservative_weight': 5.0,
                'n_action_samples': 10,
            },
            'SACPolicy': {
                
            }
        },
        'horizon': 32,
        'n_diffusion_steps': 20,
        'action_weight': 10,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 2, 4, 2, 1),
        'attention': False,
        'renderer': 'utils.MuJoCoRenderer',

        ## dataset
        'replaybuffer': 'datasets.ReplayBuffer',
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'GaussianNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 1000,

        ## serialization
        'logbase': logbase,
        'prefix': 'f:decouple_policylearning_{agent_algo}_{seed}/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 1e6,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'actor_learning_rate': 1e-5,
        'critic_learning_rate': 3e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 20000,
        'sample_freq': 20000,
        'n_saves': 5,
        'save_parallel': False,
        'bucket': None,
        'device': 'cuda',
        'seed': None,
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
        'warmup_step': 2e5,
        'actor_loss_weight': 0.001,

        'use_wandb': True,
        'group': 'default',
        'algo': 'f:{agent_algo}_diffusion', 
    },

}


#------------------------ overrides ------------------------#

walker2d_medium_replay_v2 = {
    'cost_values': {
        'termination_penalty': 1000,
    },
    'plan': {
        'n_guide_steps': 2,
    }
}

hopper_medium_expert_v2 = {
    'plan': {
        'scale': 0.0001,
        't_stopgrad': 4,
    },
}


halfcheetah_medium_replay_v2 = halfcheetah_medium_v2 = halfcheetah_medium_expert_v2 = {
    'diffusion': {
        'horizon': 4,
        'dim_mults': (1, 4, 8),
        'attention': True,
    },
    'values': {
        'horizon': 4,
        'dim_mults': (1, 4, 8),
    },
    'plan': {
        'horizon': 4,
        'scale': 0.001,
        't_stopgrad': 4,
    },
    'cost_values': {
        'horizon': 4,
        'dim_mults': (1, 4, 8),
    },
}
