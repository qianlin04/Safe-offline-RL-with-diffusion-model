from collections import namedtuple
import numpy as np
import torch
from torch import nn
import pdb

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)


Sample = namedtuple('Sample', 'trajectories values costs chains')


@torch.no_grad()
def default_sample_fn(model, x, cond, t, guide=None):
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values, values

#@torch.no_grad()
def decouple_default_sample_fn(model, x, cond, t, policy_step=True, guide=None):
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t, policy_step=policy_step)
    model_std = torch.exp(0.5 * model_log_variance)
    x_action, x_state = x[:, :, :model.action_dim], x[:, :, model.action_dim:]

    # no noise when t == 0
    noise = torch.randn_like(x_action if policy_step else x_state)
    noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values, values

def sort_by_values(x, values, costs):
    inds = torch.argsort(values, descending=True)
    x = x[inds]
    values = values[inds]
    costs = costs[inds]
    return x, values, costs

def max_by_values_in_batch(x, values, costs, sample_batch_size):
    batch_size=x.shape[0]
    x=x.reshape(sample_batch_size, batch_size//sample_batch_size, x.shape[-2], x.shape[-1])
    values=values.reshape(sample_batch_size, batch_size//sample_batch_size)
    costs=costs.reshape(sample_batch_size, batch_size//sample_batch_size)
    inds = torch.argmax(values, dim=1, keepdim=True)
    
    x=torch.gather(x, 1, inds.unsqueeze(-1).unsqueeze(-1).repeat(1,1,x.shape[-2],x.shape[-1])).squeeze(1)
    values=torch.gather(values, 1, inds).squeeze(1)
    costs=torch.gather(costs, 1, inds).squeeze(1)

    return x, values, costs


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t




class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, predict_reward_done=False, 
    ):
        super().__init__()
        self.horizon = horizon
        self.predict_reward_done = predict_reward_done
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim + 2 * predict_reward_done
        self.model = model

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=default_sample_fn, sample_batch_size=None, **sample_kwargs):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        chain = [x] if return_chain else None

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            x, values, costs = sample_fn(self, x, cond, t, **sample_kwargs)
            x = apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item(), 'cmin': costs.min().item(), 'cmax': costs.max().item()})
            if return_chain: chain.append(x)

        progress.stamp()

        if sample_batch_size is None:
            x, values, costs = sort_by_values(x, values, costs)
        else:
            x, values, costs = max_by_values_in_batch(x, values, costs, sample_batch_size)
        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x, values, costs, chain)

    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, **sample_kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, *args):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, *args, t)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)




class ValueDiffusion(GaussianDiffusion):

    def p_losses(self, x_start, cond, target, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        pred = self.model(x_noisy, cond, t)

        loss, info = self.loss_fn(pred, target)
        return loss, info

    def forward(self, x, cond, t, policy_step=True):
        return self.model(x, cond, t)



class PolicyDynamicDiffusion(nn.Module):
    def __init__(self, policy_model, dynamic_model, horizon, observation_dim, action_dim, n_timesteps=1000, 
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, sync_generate=False, predict_reward_done=False
    ):
        super().__init__()
        self.horizon = horizon
        self.predict_reward_done = predict_reward_done
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim + 2 * predict_reward_done
        self.policy_model = policy_model
        self.dynamic_model = dynamic_model

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

        self.sync_generate = sync_generate

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, policy_step):
        model = self.policy_model if policy_step else self.dynamic_model
        x_t = x[:, :, :self.action_dim] if policy_step else x[:, :, self.action_dim:]
        x_recon = self.predict_start_from_noise(x, t=t, noise=model(x, cond, t))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x_t, t=t)
        #print("p_mean_variance:         ", model(x, cond, t)[0,0])
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=decouple_default_sample_fn, **sample_kwargs):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        chain = [x] if return_chain else None

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            
            x_action, values, costs = sample_fn(self, x, cond, t, policy_step=True, **sample_kwargs)
            if not self.sync_generate:
                x_dynamic_input = torch.cat([x_action, x[:, :, self.action_dim:]], -1)
                x_dynamic_input = apply_conditioning(x_dynamic_input, cond, self.action_dim)
            else:
                x_dynamic_input = x
              
            x_state, _, costs = sample_fn(self, x_dynamic_input, cond, t, policy_step=False, **sample_kwargs)
            x = torch.cat([x_action, x_state], -1)
            x = apply_conditioning(x, cond, self.action_dim)
            progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item(), 'cmin': costs.min().item(), 'cmax': costs.max().item()})
            if return_chain: chain.append(x)

        progress.stamp()

        x, values, costs = sort_by_values(x, values, costs)
        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x, values, costs, chain)

    def p_sample_loop_for_train(self, shape, cond, sample_fn=decouple_default_sample_fn, **sample_kwargs):
        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
         
            x_action, values, costs = sample_fn(self, x, cond, t, policy_step=True, **sample_kwargs)
            if not self.sync_generate:
                x_dynamic_input = torch.cat([x_action, x[:, :, self.action_dim:]], -1)
                x_dynamic_input = apply_conditioning(x_dynamic_input, cond, self.action_dim)
            else:
                x_dynamic_input = x
              
            x_state, _, costs = sample_fn(self, x_dynamic_input, cond, t, policy_step=False, **sample_kwargs)
            x = torch.cat([x_action, x_state], -1)
            x = apply_conditioning(x, cond, self.action_dim)
        return Sample(x, values, costs, None)

    def conditional_sample(self, cond, horizon=None, no_grad=True, training=False, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        p_sample_loop = self.p_sample_loop_for_train if training else self.p_sample_loop
        if no_grad:
            with torch.no_grad():
                return p_sample_loop(shape, cond, **sample_kwargs)
        else:
            return p_sample_loop(shape, cond, **sample_kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, torch.abs(t), x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, torch.abs(t), x_start.shape) * noise
        )
        sample = torch.where(t.unsqueeze(-1).unsqueeze(-1).expand_as(x_start)==-1, x_start, sample)
        return sample

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)
        
        if not self.sync_generate:
            x_noisy_action = x_noisy[:, :, :self.action_dim]
            x_start_action = x_start[:, :, :self.action_dim]
            model_log_variance = extract(self.posterior_log_variance_clipped, t, x_noisy_action.shape)
            model_std = torch.exp(0.5 * model_log_variance)
            model_mean, _, _ = self.q_posterior(x_start=x_start_action, x_t=x_noisy_action, t=t)
            action_noise = torch.randn_like(x_noisy_action)
            action_noise[t == 0] = 0
            new_action = model_mean + model_std * action_noise
            x_dynamic_input = torch.cat([new_action, x_noisy[:, :, self.action_dim:]], -1) # x_noisy
            # x_noisy_2 = self.q_sample(x_start=x_start, t=t-1)
            # x_dynamic_input = torch.cat([x_noisy_2[:, :, :self.action_dim], x_noisy[:, :, self.action_dim:]], -1) # x_noisy
        else:
            x_dynamic_input = x_noisy

        action_recon = self.policy_model(x_noisy, cond, t)
        state_recon = self.dynamic_model(x_dynamic_input, cond, t)
        x_recon = torch.cat([action_recon, state_recon], -1)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        #print(x_start[0, 0, :self.action_dim], x_recon[0, 0, :self.action_dim])

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        info['action_recon'] = action_recon
        info['state_recon'] = state_recon

        return loss, info

    def loss(self, x, *args):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, *args, t)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)


class DecoupleValueDiffusion(PolicyDynamicDiffusion):

    def p_losses(self, x_start, cond, target, t):
        noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)
        
        if not self.sync_generate:
            x_noisy_action = x_noisy[:, :, :self.action_dim]
            x_start_action = x_start[:, :, :self.action_dim]
            model_log_variance = extract(self.posterior_log_variance_clipped, t, x_noisy_action.shape)
            model_std = torch.exp(0.5 * model_log_variance)
            model_mean, _, _ = self.q_posterior(x_start=x_start_action, x_t=x_noisy_action, t=t)
            action_noise = torch.randn_like(x_noisy_action)
            action_noise[t == 0] = 0
            new_action = model_mean + model_std * action_noise
            x_dynamic_input = torch.cat([new_action, x_noisy[:, :, self.action_dim:]], -1) # x_noisy
            # x_noisy_2 = self.q_sample(x_start=x_start, t=t-1)
            # x_dynamic_input = torch.cat([x_noisy_2[:, :, :self.action_dim], x_noisy[:, :, self.action_dim:]], -1) # x_noisy
        else:
            x_dynamic_input = x_noisy

        pred_policy_step = self.policy_model(x_noisy, cond, t)
        pred_dynamic_step = self.dynamic_model(x_dynamic_input, cond, t)

        loss1, info1 = self.loss_fn(pred_policy_step, target)
        loss2, info2 = self.loss_fn(pred_dynamic_step, target)
        loss = loss1 + loss2
        info = {k:(info1[k]+info2[k])*0.5 for k in info1}
        return loss, info

    def forward(self, x, cond, t, policy_step=True):
        return self.policy_model(x, cond, t) if policy_step else self.dynamic_model(x, cond, t)