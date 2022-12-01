import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Box, Discrete 

from copy import deepcopy
from diffuser.agent.SACPolicy import SACPolicy

class CQLPolicy(SACPolicy, nn.Module):
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
            action_normalizer=action_normalizer,
        )
        self.conservative_weight = conservative_weight
        self.n_action_samples = n_action_samples

    def sample_random_action(self, batch_size):
        if type(self.action_space) is Box:
            action = np.random.uniform(self.action_space.low, self.action_space.high, size=(batch_size, len(self.action_space.low)))
        elif type(self.action_space) is Discrete:
            action = np.eye(self.action_space.n)[np.random.randint(0, self.action_space.n, size=batch_size)]
        else:
            action = [self.action_space.sample() for _ in range(batch_size)]
            action = np.vstack(action)
        action = self.action_normalizer.normalize(action)
        action = torch.as_tensor(action).to(self._device)
        return action

    def compute_q_value(self, policy_obs, value_obs, random_policy=False):
        obs_shape = policy_obs.shape
        policy_obs = policy_obs.expand(self.n_action_samples, *obs_shape)
        policy_obs = policy_obs.transpose(0, 1)
        policy_obs = policy_obs.reshape(-1, *obs_shape[1:])
        if random_policy:
            policy_actions = self.sample_random_action(policy_obs.shape[0]) 
        else:
            with torch.no_grad():
                policy_actions, _ = self(policy_obs)

        obs_shape = value_obs.shape
        repeated_obs = value_obs.expand(self.n_action_samples, *obs_shape)
        # (n, batch, observation) -> (batch, n, observation)
        transposed_obs = repeated_obs.transpose(0, 1)
        # (batch, n, observation) -> (batch * n, observation)
        flat_obs = transposed_obs.reshape(-1, *obs_shape[1:])
        # (batch, n, action) -> (batch * n, action)
        flat_policy_acts = policy_actions.reshape(-1, self.action_dim)

        # estimate action-values for policy actions
        policy_values_1 = self.critic1(flat_obs, flat_policy_acts)
        policy_values_2 = self.critic2(flat_obs, flat_policy_acts)
        policy_values_1 = policy_values_1.view(
            obs_shape[0], self.n_action_samples
        )
        policy_values_2 = policy_values_2.view(
            obs_shape[0], self.n_action_samples
        )

        return policy_values_1, policy_values_2

    def compute_conservative_loss(self, obs_t: torch.Tensor, act_t: torch.Tensor, obs_tp1: torch.Tensor):
        policy_values_t_1, policy_values_t_2 = self.compute_q_value(obs_t, obs_t)
        policy_values_tp1_1, policy_values_tp1_2 = self.compute_q_value(obs_tp1, obs_t)
        random_values_1, random_values_2 = self.compute_q_value(obs_t, obs_t, random_policy=True)

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
        data_values_1 = self.critic1(obs_t, act_t)
        data_values_2 = self.critic2(obs_t, act_t)

        loss_1 = logsumexp_1.mean(dim=0).mean() - data_values_1.mean(dim=0).mean()
        scaled_loss_1 = self.conservative_weight * loss_1
        loss_2 = logsumexp_2.mean(dim=0).mean() - data_values_2.mean(dim=0).mean()
        scaled_loss_2 = self.conservative_weight * loss_2

        return scaled_loss_1, scaled_loss_2

    def update_critic(self, obs, actions, next_obs, terminals, rewards):
        # compute conservative loss
        conservative_loss_1, conservative_loss_2 = self.compute_conservative_loss(obs, actions, next_obs)

        # update critic
        q1, q2 = self.critic1(obs, actions).flatten(), self.critic2(obs, actions).flatten()
        with torch.no_grad():
            next_actions, _ = self(next_obs)
            next_q = torch.min(
                self.critic1_old(next_obs, next_actions), self.critic2_old(next_obs, next_actions)
            )
            target_q = rewards.flatten() + self._gamma * (1 - terminals.flatten()) * next_q.flatten()
        critic1_td_loss = ((q1 - target_q).pow(2)).mean()
        critic2_td_loss = ((q2 - target_q).pow(2)).mean()
        critic1_loss = critic1_td_loss + conservative_loss_1
        critic2_loss = critic2_td_loss + conservative_loss_2
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()
        return critic1_loss, critic2_loss

    def learn(self, data):
        obs, actions, next_obs, terminals, rewards = data["observations"], \
            data["actions"], data["next_observations"], data["terminals"], data["rewards"]
        
        obs = torch.as_tensor(obs).to(self._device)
        actions = torch.as_tensor(actions).to(self._device)
        next_obs = torch.as_tensor(next_obs).to(self._device)
        rewards = torch.as_tensor(rewards).to(self._device)
        terminals = torch.as_tensor(terminals).to(self._device)

        critic1_loss, critic2_loss = self.update_critic(obs, actions, next_obs, terminals, rewards)
        actor_loss = self.update_actor(obs)
        self._sync_weight()

        result =  {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item()
        }

        # if self._is_auto_alpha:
        #     result["loss/alpha"] = alpha_loss.item()
        #     result["alpha"] = self._alpha.item()
        
        return result
