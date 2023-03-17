import diffuser.utils as utils


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'hopper-medium-expert-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('diffusion')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
    predict_reward_done=args.predict_reward_done, 
)

render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    env=args.dataset,
)

dataset = dataset_config()
renderer = render_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim
transition_dim = observation_dim + action_dim + 2 * args.predict_reward_done
 
#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#


if "decouple" in args.config:
    dynamic_model_config = utils.Config(
        args.model,
        savepath=(args.savepath, 'dynamic_model_config.pkl'),
        horizon=args.horizon,
        transition_dim=transition_dim,
        output_dim=observation_dim + 2 * args.predict_reward_done,
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
else:
    model_config = utils.Config(
        args.model,
        savepath=(args.savepath, 'model_config.pkl'),
        horizon=args.horizon,
        transition_dim=transition_dim,
        cond_dim=observation_dim,
        dim_mults=args.dim_mults,
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
    predict_reward_done = args.predict_reward_done, 
)

trainer_config = utils.Config(
    utils.Trainer,
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
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

if "decouple" in args.config:
    dynamic_model = dynamic_model_config()
    policy_model = policy_model_config()
    diffusion = diffusion_config(policy_model, dynamic_model)
    utils.report_parameters(dynamic_model)
    utils.report_parameters(policy_model)
else:
    model = model_config()
    diffusion = diffusion_config(model)
    utils.report_parameters(model)

trainer = trainer_config(diffusion, dataset, renderer)

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

