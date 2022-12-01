import diffuser.utils as utils
import numpy as np
import gym
import gym.spaces
import d4rl
import wandb

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser): 
    dataset: str = 'hopper-medium-expert-v2'
    config: str = 'config.locomotion'
    project: str = 'DecoupleDifussion'

args = Parser().parse_args('policylearning')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

if args.use_wandb:
    wandb.init(
        project=args.project,
        config=args,
        name=f'{args.dataset}_{args.algo}_{args.seed}',
        group=args.group
    )

env = gym.make(args.dataset)
dataset = d4rl.qlearning_dataset(env)

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
    predict_reward_done=True, 
)

replaybuffer_config = utils.Config(
    args.replaybuffer,
    obs_shape=env.observation_space.shape if not type(env.observation_space) is gym.spaces.Discrete else (env.observation_space.n, ),
    obs_dtype=np.float32,
    action_dim=int(np.prod(env.action_space.shape)) if not type(env.action_space) is gym.spaces.Discrete else env.action_space.n,
    action_dtype=np.float32
)

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    env=args.dataset,
)

offline_buffer = replaybuffer_config(buffer_size=1)
offline_buffer.load_dataset(dataset, normalizer=args.normalizer)
model_buffer = replaybuffer_config(buffer_size=args.rollout_batch_size*args.rollout_length*args.model_retain_epochs)
dataset = dataset_config()
renderer = render_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim
transition_dim = observation_dim + action_dim + 2

#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

dynamic_model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'dynamic_model_config.pkl'),
    horizon=args.horizon,
    transition_dim=transition_dim,
    output_dim=observation_dim + 2,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    attention=args.attention,
    device=args.device,
)

policy_model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'policy_model_config.pkl'),
    horizon=args.horizon,
    transition_dim=transition_dim,
    output_dim=action_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    attention=args.attention,
    device=args.device,
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    ## loss weighting
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
    predict_reward_done=True,
)

agent_config = utils.Config(
    args.agent,
    critic_input_dim=observation_dim+action_dim, 
    actor_lr=args.actor_learning_rate,
    critic_lr=args.critic_learning_rate,
    to_device=args.device,
    device=args.device,
    action_space=env.action_space,
    **args.agent_params,
)


trainer_config = utils.Config(
    utils.AgentTrainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
    rollout_length=args.rollout_length,
    rollout_batch_size=args.rollout_batch_size, 
    rollout_freq=args.rollout_freq,
    train_policy_freq=args.train_policy_freq,
    real_ratio=args.real_ratio,
    agent_batch_size=args.agent_batch_size,
    use_wandb=args.use_wandb,
    warmup_step=args.warmup_step,
    action_normalizer=offline_buffer.normalizers['action'] if offline_buffer.normalizers is not None else None
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

dynamic_model = dynamic_model_config()
policy_model = policy_model_config()
diffusion = diffusion_config(policy_model, dynamic_model)
utils.report_parameters(dynamic_model)
utils.report_parameters(policy_model)
agent = agent_config(diffusion)

trainer = trainer_config(diffusion, dataset, renderer, agent, offline_buffer, model_buffer)
if hasattr(args, 'load_epoch') and args.load_epoch:
    trainer.load(args.load_epoch)

#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#


print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0])
loss, _ = diffusion.loss(*batch)
loss.backward()
print('✓')


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)
    eval_results = trainer.evaluate(env, eval_episodes=args.eval_episodes)
    print(eval_results)

