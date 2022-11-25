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
    n_test_episode: int = 5

args = Parser().parse_args('plan')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

## load diffusion model and value function from disk
load_diffusion_func = utils.load_diffusion if not "decouple" in args.diffusion_loadpath else utils.load_decouple_diffusion
diffusion_experiment = load_diffusion_func(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)
load_diffusion_func = utils.load_diffusion if not "decouple" in args.value_loadpath else utils.load_decouple_diffusion
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
    sample_fn=sampling.n_step_guided_p_sample if not "decouple" in args.config else sampling.n_step_guided_p_sample_decouple,
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
all_return = []
for _ in range(args.n_test_episode):

    env = dataset.env
    observation = env.reset()

    ## observations for rendering
    rollout = [observation.copy()]

    total_reward = 0

    for t in range(args.max_episode_length):

        #if t % 10 == 0: print(args.savepath, flush=True)

        ## save state for rendering only
        # state = env.state_vector().copy()

        ## format current observation for conditioning
        conditions = {0: observation}
        action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)

        ## execute action in environment
        next_observation, reward, terminal, _ = env.step(action)
        
        # import numpy as np
        # env.render()
        # dire = ['U', 'R', 'D', 'L']
        # print("action:   ", dire[np.argmax(action)], np.max(action))
        # for k, traj in enumerate(samples.observations):
        #     o = np.zeros((4,12))
        #     for i in range(8):
        #         pos = np.argmax(traj[i, :])
        #         o[ np.unravel_index(pos, o.shape)] = i+1
        #     print(o)
        #     print("values:  ", samples.values[k])
        # input()

        ## print reward and score
        total_reward += reward
        if hasattr(env, 'get_normalized_score'):
            score = env.get_normalized_score(total_reward)
        else:
            score = 0
        print(
            f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
            f'values: {samples.values[0]} | scale: {args.scale}',
            flush=True,
        )

        ## update rollout observations
        rollout.append(next_observation.copy())

        ## render every `args.vis_freq` steps
        # logger.log(t, samples, state, rollout)

        if terminal:
            break

        observation = next_observation

    all_results.append(f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
                       f'values: {samples.values[0]} | scale: {args.scale}')
    all_scores.append(score)
    all_return.append(total_reward)

    ## write results to json file at `args.savepath`
    #logger.finish(t, score, total_reward, terminal, diffusion_experiment, value_experiment)
    for res in all_results:
        print(res)
    print(all_scores, all_return)
    import numpy as np
    print("scores:      ", np.mean(all_scores))
    print("return:      ", np.mean(all_return))