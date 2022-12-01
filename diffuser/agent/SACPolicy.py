import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=None, activation=nn.ReLU):
        super().__init__()
        hidden_dims = [input_dim] + list(hidden_dims)
        model = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            model += [nn.Linear(in_dim, out_dim), activation()]
        
        self.output_dim = hidden_dims[-1]
        if output_dim is not None:
            model += [nn.Linear(hidden_dims[-1], output_dim)]
            self.output_dim = output_dim
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=None, activation=nn.ReLU, device="cpu"):
        super().__init__()
        backbone = MLP(input_dim, hidden_dims, output_dim=output_dim, activation=activation)
        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        latent_dim = getattr(backbone, "output_dim")
        self.last = nn.Linear(latent_dim, 1).to(device)

    def forward(self, obs, actions=None):
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if actions is not None:
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32).flatten(1)
            obs = torch.cat([obs, actions], dim=1)
        logits = self.backbone(obs)
        values = self.last(logits)
        return values



class SACPolicy(nn.Module):
    def __init__(
        self, 
        actor, 
        critic_input_dim, 
        actor_lr,
        critic_lr,
        action_space,
        critic_hidden_dims = [256, 256], 
        tau=0.5, 
        gamma=0.99, 
        alpha=0.2,
        to_device="cpu",
        action_normalizer=None,
    ):
        super().__init__()

        self.action_space = action_space
        self.action_dim = actor.action_dim
        self.observation_dim = actor.observation_dim
        self.actor = actor
        self.actor_lr = actor_lr
        critic1 = Critic(input_dim=critic_input_dim, hidden_dims=critic_hidden_dims, device=to_device)
        actor_optim = torch.optim.Adam(actor.policy_model.parameters(), lr=actor_lr)
        critic2 = Critic(input_dim=critic_input_dim, hidden_dims=critic_hidden_dims, device=to_device)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)

        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()

        self.actor_optim = actor_optim
        self.critic1_optim = critic1_optim
        self.critic2_optim = critic2_optim
        
        self._tau = tau
        self._gamma = gamma

        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self._alpha_optim = alpha
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha
        
        self.__eps = np.finfo(np.float32).eps.item()

        self.action_normalizer = action_normalizer
        self._device = to_device

    def load_actor(self, actor):
        self.actor = actor
        self.actor_optim = torch.optim.Adam(actor.policy_model.parameters(), lr=self.actor_lr)

    def train(self):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
    
    def _sync_weight(self):
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    def forward(self, obs, horizon=1, no_grad=True, deterministic=False):
        cond = {0: torch.tensor(obs, device=self._device)}
        samples = self.actor(cond, no_grad=no_grad, horizon=horizon, training=True)
        trajectories = samples.trajectories
        action = trajectories[:, 0, :self.action_dim]

        # dire = ['U', 'R', 'D', 'L']
        # tr = trajectories[:, :, self.action_dim:self.action_dim+self.observation_dim].cpu().numpy()
        # print("action:   ", dire[np.argmax(action.cpu().numpy())], np.max(action.cpu().numpy()))
        # print(obs[0], np.where(obs[0]!=0)[0])
        # print(action)
        # for k, traj in enumerate(tr):
        #     o = np.zeros((4,12))
        #     for i in range(8):
        #         pos = np.argmax(traj[i, :])
        #         o[ np.unravel_index(pos, o.shape)] = i+1
        #     print(o)
        # input()

        return action, trajectories

    def get_transitions(self, trajectories):
        tran_dim = self.action_dim + self.observation_dim
        actions = trajectories[:, :, :self.action_dim].cpu().numpy()
        observations = trajectories[:, :, self.action_dim:tran_dim].cpu().numpy()
        rewards = trajectories[:, :, tran_dim:tran_dim+1].cpu().numpy()
        terminals = trajectories[:, :, tran_dim+1:].cpu().numpy()
        return observations, actions, rewards, terminals

    def sample_action(self, obs, horizon=None, deterministic=False):
        obs = np.expand_dims(obs, 0)
        actions, trajectories = self(obs, horizon=horizon)
        return actions[0].cpu().detach().numpy()

    def update_critic(self, obs, actions, next_obs, terminals, rewards):
        # update critic
        q1, q2 = self.critic1(obs, actions).flatten(), self.critic2(obs, actions).flatten()
        with torch.no_grad():
            next_actions, _ = self(next_obs)
            next_q = torch.min(
                self.critic1_old(next_obs, next_actions), self.critic2_old(next_obs, next_actions)
            )
            target_q = rewards.flatten() + self._gamma * (1 - terminals.flatten()) * next_q.flatten()
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()
        critic2_loss = ((q2 - target_q).pow(2)).mean()
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()
        return critic1_loss, critic2_loss

    def update_actor(self, obs):
        # update actor
        a, _ = self(obs, no_grad=False)
        q1a, q2a = self.critic1(obs, a).flatten(), self.critic2(obs, a).flatten()
        actor_loss = - torch.min(q1a, q2a).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        return actor_loss

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
