from builtins import all
import pdb

import diffuser.sampling as sampling
import diffuser.utils as utils


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'walker2d-medium-replay-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('plan')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model and value function from disk
load_diffusion_func = utils.load_diffusion if not "decouple" in args.config else utils.load_decouple_diffusion
diffusion_experiment = load_diffusion_func(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)
value_experiment = load_diffusion_func(
    args.loadbase, args.dataset, args.value_loadpath,
    epoch=args.value_epoch, seed=args.seed,
)

## ensure that the diffusion model and value function are compatible with each other
utils.check_compatibility(diffusion_experiment, value_experiment)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

## initialize value guide
value_function = value_experiment.ema
guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()

logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)

## policies are wrappers around an unconditional diffusion model and a value guide
policy_config = utils.Config(
    args.policy,
    guide=guide,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    ## sampling kwargs
    sample_fn=sampling.n_step_guided_p_sample if not "decouple" in args.diffusion_loadpath else sampling.n_step_guided_p_sample_decouple,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    state_grad_mask=args.state_grad_mask,
    verbose=False,
)

logger = logger_config()
policy = policy_config()


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#
all_results = []
all_scores = []
for _ in range(1):

    env = dataset.env
    env._max_episode_steps = args.max_episode_length = 2000
    observation = env.reset([1, 0, 0, 0])

    ## observations for rendering
    rollout = [observation.copy()]

    total_reward = 0

    for t in range(args.max_episode_length):

        
        ## format current observation for conditioning
        conditions = {0: observation}
        action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)

        import numpy as np
        # for i in range(len(samples.observations)):
        #     n = [np.sum(np.array([1,2,3,4])*(samples.observations[i,j]>0.7)) for j in range(4)]
        #     print(n, "   |   ", samples.actions[i].squeeze())

        # input()

        ## execute action in environment
        next_observation, reward, terminal, _ = env.step(action)

        print(observation,0 if action<0.5 else 1)

        ## print reward and score
        total_reward += reward
        score = 0
        print(
            f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
            f'values: {samples.values[0]} | scale: {args.scale}',
            flush=True,
        )

        ## update rollout observations
        rollout.append(next_observation.copy())

        
        if terminal:
            break

        observation = next_observation

    all_results.append(f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
                       f'values: {samples.values[0]} | scale: {args.scale}')
    all_scores.append(score)

    ## write results to json file at `args.savepath`
    #logger.finish(t, score, total_reward, terminal, diffusion_experiment, value_experiment)
    for res in all_results:
        print(res)
    print(all_scores)
    import numpy as np
    print(np.mean(all_scores))
