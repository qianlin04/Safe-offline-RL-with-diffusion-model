import os
import copy
import numpy as np
import torch
import einops
import pdb
import wandb

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        total_loss = [] 
        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch)

                loss, infos = self.model.loss(*batch)
                total_loss.append(loss.cpu().item())
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}', flush=True)

            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference)

            if self.sample_freq and self.step % self.sample_freq == 0:
                self.render_samples()

            self.step += 1
        print("mean loss : ", np.mean(total_loss))

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim:self.dataset.action_dim+self.dataset.observation_dim]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        if self.renderer is not None: 
            self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, 'cuda:0')

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model(conditions)
            trajectories = to_np(samples.trajectories)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = trajectories[:, :, self.dataset.action_dim:self.dataset.action_dim+self.dataset.observation_dim]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            if self.renderer is not None: 
                self.renderer.composite(savepath, observations)



class AgentTrainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        agent=None,
        offline_buffer=None,
        model_buffer=None,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,

        warmup_step=2e5,
        rollout_length=5,
        rollout_batch_size=50000,
        rollout_freq=1000,
        train_policy_freq=1,
        real_ratio=0.05,
        agent_batch_size=256,
        use_wandb=False,
    ):
        super().__init__()
        self.model = diffusion_model
        self.agent = agent
        self.dataset = dataset
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.offline_buffer = offline_buffer
        self.model_buffer = model_buffer

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference

        self.reset_parameters()
        self.step = 0
        self.warmup_step = warmup_step
        self.rollout_length = rollout_length
        self.rollout_batch_size = rollout_batch_size
        self.rollout_freq = rollout_freq
        self.real_ratio = real_ratio
        self.agent_batch_size = agent_batch_size
        self.train_policy_freq = train_policy_freq
        self.use_wandb = use_wandb

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def rollout_transitions(self, ):
        init_transitions = self.offline_buffer.sample(batch_size=self.rollout_batch_size)
        init_observations = init_transitions['observations']
        actions, samples = self.agent(init_observations, horizon=self.rollout_length)
        observations, actions, rewards, terminals = self.agent.get_transitions(samples)
        terminals = terminals > 0.5
        nonterm_mask = np.ones(self.rollout_batch_size, dtype=np.bool_)
        for h in range(observations.shape[1]-1):
            observation = observations[:, h, :][nonterm_mask]
            action = actions[:, h, :][nonterm_mask]
            next_observation = observations[:, h+1, :][nonterm_mask]
            reward = rewards[:, h, :][nonterm_mask]
            terminal = terminals[:, h, :][nonterm_mask] 
            nonterm_mask = np.logical_and(1-terminals[:, h, 0], nonterm_mask)
            self.model_buffer.add_batch(observation, next_observation, action, reward, terminal)
            if nonterm_mask.sum() == 0:
                break

        # for k, traj in enumerate(observations[:5]):
        #     print('---------------------------------------')
        #     o = np.zeros((4,12))
        #     for i in range(self.rollout_length):
        #         pos = np.argmax(traj[i, :])
        #         o[ np.unravel_index(pos, o.shape)] = i+1
        #         if terminals[k, i, 0]==True:
        #             o[ np.unravel_index(pos, o.shape)] = -1
        #             break
        #     print(o)
        #     print(-np.sort(-observations[k, :, :], axis=-1)[:,:3], -np.sort(-actions[k, :, :], axis=-1)[:, :3])


    def train_diffusion(self, ):
        for i in range(self.gradient_accumulate_every):
            batch = next(self.dataloader)
            batch = batch_to_device(batch)
            loss, infos = self.model.loss(*batch)
            if self.use_wandb:
                wandb.log({"loss/diffusion/loss": loss.cpu().item()}, step=self.step) 
                for k, v in infos.items():
                    wandb.log({f"loss/diffusion/{k}": v}, step=self.step) 
            loss = loss / self.gradient_accumulate_every
            loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.cpu().item()

    def train_policy(self,):
        # update policy by sac
        real_sample_size = int(self.agent_batch_size * self.real_ratio)
        fake_sample_size = self.agent_batch_size - real_sample_size
        real_batch = self.offline_buffer.sample(batch_size=real_sample_size)
        fake_batch = self.model_buffer.sample(batch_size=fake_sample_size)
        data = {
            "observations": np.concatenate([real_batch["observations"], fake_batch["observations"]], axis=0),
            "actions": np.concatenate([real_batch["actions"], fake_batch["actions"]], axis=0),
            "next_observations": np.concatenate([real_batch["next_observations"], fake_batch["next_observations"]], axis=0),
            "terminals": np.concatenate([real_batch["terminals"], fake_batch["terminals"]], axis=0),
            "rewards": np.concatenate([real_batch["rewards"], fake_batch["rewards"]], axis=0)
        }
        results = self.agent.learn(data)
        if self.use_wandb:
            wandb.log({"loss/actor": results["loss/actor"]}, step=self.step)
            wandb.log({"loss/critic1": results["loss/critic1"]}, step=self.step)
            wandb.log({"loss/critic2": results["loss/critic2"]}, step=self.step)
        return results

    def train(self, n_train_steps):
        self.agent.train()
        timer = Timer()
        for step in range(n_train_steps):
            loss = self.train_diffusion()

            if step % self.rollout_freq==0 and self.step>=self.warmup_step == 0:
                self.rollout_transitions()

            if step % self.train_policy_freq==0 and self.step>=self.warmup_step == 0:
                self.train_policy()
            
            # if step%100==0:
            #     print("--------------------------------")
            #     qv = np.zeros((4,12))
            #     qa = np.zeros((4,12))
            #     policy_a = np.zeros((4,12,4))
            #     dire = ['U', 'R', 'D', 'L'] #
            #     for i in range(int(np.prod(qv.shape))):
            #         obs = torch.zeros((4, np.prod(qv.shape)), device="cuda")
            #         obs[:, i] = 1
            #         a = torch.eye(4, device="cuda")
            #         with torch.no_grad():
            #             q1a, q2a = self.agent.critic1_old(obs, a).flatten(), self.agent.critic2_old(obs, a).flatten()
            #             q = torch.min(q1a, q2a)
            #         a,b = np.unravel_index(i, qv.shape)
            #         qv[a, b]=q.max()
            #         qa[a, b]=q.argmax()

            #         s = 256
            #         obs = torch.zeros((s, np.prod(qv.shape)), device="cuda")
            #         obs[:, i] = 1
            #         pa, _ = self.agent(obs, horizon=1)
            #         pa = pa.cpu().numpy()
            #         pa = np.argmax(pa, axis=-1)
            #         for j in range(4):
            #             policy_a[a, b, j] = np.sum(pa==j) / s

            #     print("q_table")
            #     for i in range(4):
            #         p = " ".join([str(j)[:6] for j in qv[i]])
            #         print(p)
            #     print("q_table_action")
            #     for i in range(4):
            #         p = " ".join([dire[int(j)] for j in qa[i]])
            #         print(p)
            #     print("policy_action")
            #     for i in range(4):
            #         p = ""
            #         for a in policy_a[i]:
            #             for kk, k in enumerate(np.argsort(-a)[:2]):
            #                 if kk!=0: p+=","
            #                 if a[k]<0.1:
            #                     p+="     "
            #                 else:
            #                     p+=dire[int(k)]+":%.1f" % a[k]

            #             p += " "
            #         print(p)
            #     print("--------------------------------")
            #     print()
            
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.log_freq == 0:
                #infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.4f} | t: {timer():8.4f}', flush=True)

            self.step += 1
        #print("mean loss : ", np.mean(total_loss))

    def evaluate(self, eval_env, eval_episodes=10):
        self.agent.eval()
        obs = eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < eval_episodes:
            normed_obs = self.offline_buffer.normalize(obs)
            normed_action = self.agent.sample_action(normed_obs,)
            action = self.offline_buffer.unnormalize(normed_action)
            next_obs, reward, terminal, _ = eval_env.step(action)
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if terminal:
                score = eval_env.get_normalized_score(episode_reward) if hasattr(eval_env, 'get_normalized_score') else 0
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length, 'score': score}
                )
                num_episodes += 1
                episode_reward, episode_length = 0, 0
                obs = eval_env.reset()
        
        if self.use_wandb:
            wandb.log({"eval/score": np.mean([ep_info["score"] for ep_info in eval_ep_info_buffer])}, step=self.step)
            wandb.log({"eval/episode_reward": np.mean([ep_info["episode_reward"] for ep_info in eval_ep_info_buffer])}, step=self.step)
            wandb.log({"eval/episode_length": np.mean([ep_info["episode_length"] for ep_info in eval_ep_info_buffer])}, step=self.step)

        return {
            "eval/score": [ep_info["score"] for ep_info in eval_ep_info_buffer],
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        if self.agent is not None: 
            self.agent.load_actor(self.model)

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim:self.dataset.action_dim+self.dataset.observation_dim]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        if self.renderer is not None: 
            self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, 'cuda:0')

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model(conditions)
            trajectories = to_np(samples.trajectories)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = trajectories[:, :, self.dataset.action_dim:self.dataset.action_dim+self.dataset.observation_dim]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            if self.renderer is not None: 
                self.renderer.composite(savepath, observations)