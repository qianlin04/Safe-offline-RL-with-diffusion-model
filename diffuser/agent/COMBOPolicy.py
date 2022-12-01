import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Box, Discrete 

from copy import deepcopy
from diffuser.agent.CQLPolicy import CQLPolicy

class COMBOPolicy(CQLPolicy, nn.Module):
    def __init__(
        self, 
        actor, 
        critic_input_dim, 
        actor_lr,
        critic_lr,
        action_space, 
        critic_hidden_dims=[256, 256], 
        tau=0.5, 
        gamma=0.99, 
        alpha=0.2,
        to_device="cpu",
        conservative_weight=5.0,
        n_action_samples=10,
        real_ratio=1.0,
        action_normalizer=None,
    ):
        super().__init__(
            actor=actor, 
            critic_input_dim=critic_input_dim, 
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            action_space=action_space,
            critic_hidden_dims=critic_hidden_dims, 
            tau=tau, 
            gamma=gamma, 
            alpha=alpha,
            to_device=to_device,
            conservative_weight=conservative_weight,
            n_action_samples=n_action_samples,
            action_normalizer=action_normalizer,
        )
        self.real_ratio = real_ratio


    def compute_conservative_loss(self, obs_t: torch.Tensor, act_t: torch.Tensor, obs_tp1: torch.Tensor):

        fake_obs_t = obs_t[int(obs_t.shape[0] * self.real_ratio) :]
        fake_obs_tp1 = obs_tp1[int(obs_tp1.shape[0] * self.real_ratio) :]
        real_obs_t = obs_t[: int(obs_t.shape[0] * self.real_ratio)]
        real_act_t = act_t[: int(act_t.shape[0] * self.real_ratio)]

        policy_values_t_1, policy_values_t_2 = self.compute_q_value(fake_obs_t, fake_obs_t)
        policy_values_tp1_1, policy_values_tp1_2 = self.compute_q_value(fake_obs_tp1, fake_obs_t)
        random_values_1, random_values_2 = self.compute_q_value(fake_obs_t, fake_obs_t, random_policy=True)

        # compute logsumexp
        # (batch, 3 * n samples) -> (batch, 1)
        target_values_1 = torch.cat(
            [policy_values_t_1, policy_values_tp1_1, random_values_1], dim=1
        )
        target_values_2 = torch.cat(
            [policy_values_t_2, policy_values_tp1_2, random_values_2], dim=1
        )
        logsumexp_1 = torch.logsumexp(target_values_1, dim=1, keepdim=True)
        logsumexp_2 = torch.logsumexp(target_values_2, dim=1, keepdim=True)

        # estimate action-values for data actions
        data_values_1 = self.critic1(real_obs_t, real_act_t)
        data_values_2 = self.critic2(real_obs_t, real_act_t)

        loss_1 = logsumexp_1.mean(dim=0).mean() - data_values_1.mean(dim=0).mean()
        scaled_loss_1 = self.conservative_weight * loss_1
        loss_2 = logsumexp_2.mean(dim=0).mean() - data_values_2.mean(dim=0).mean()
        scaled_loss_2 = self.conservative_weight * loss_2

        return scaled_loss_1, scaled_loss_2

