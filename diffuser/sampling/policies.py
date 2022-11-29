from collections import namedtuple
import torch
import einops
import pdb
import numpy as np

import diffuser.utils as utils
from diffuser.datasets.preprocessing import get_policy_preprocess_fn


Trajectories = namedtuple('Trajectories', 'actions observations rewards terminals values costs')


class GuidedPolicy:

    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        self.guide = guide
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.observation_dim = diffusion_model.observation_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.sample_kwargs = sample_kwargs

    def __call__(self, conditions, batch_size=1, verbose=True, cost_threshold=None):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)

        ## run reverse diffusion process
        if cost_threshold is not None: self.sample_kwargs['cost_threshold'] = cost_threshold        #change the cost threshold dynamically
        samples = self.diffusion_model(conditions, guide=self.guide, verbose=verbose, **self.sample_kwargs)
        trajectories = utils.to_np(samples.trajectories)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = trajectories[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')

        ## extract first action
        action = actions[0, 0]

        tran_dim = self.action_dim+self.observation_dim
        normed_observations = trajectories[:, :, self.action_dim:tran_dim]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
        
        if trajectories.shape[-1] > tran_dim:
            normed_rewards = trajectories[:, :, tran_dim:tran_dim+1]
            normed_terminals = trajectories[:, :, tran_dim+1:tran_dim+2]
            rewards = self.normalizer.unnormalize(normed_rewards, 'rewards')
            terminals = self.normalizer.unnormalize(normed_terminals, 'terminals')
        else:
            rewards = np.zeros(trajectories.shape[:-1]+[1])
            terminals = np.zeros(trajectories.shape[:-1]+[1])

        trajectories = Trajectories(actions, observations, rewards, terminals, samples.values, samples.costs)
        return action, trajectories

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device='cuda:0')
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions
