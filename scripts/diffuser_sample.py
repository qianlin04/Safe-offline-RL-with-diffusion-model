from builtins import all
import pdb

import diffuser.sampling as sampling
import diffuser.utils as utils


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = "TwoStepMDP-v0"
    config: str = 'two_step_mdp_experiment.config'

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

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

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
    guide=None,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
)

logger = logger_config()
policy = policy_config()


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#
all_results = []
all_scores = []
for _ in range(5):

    env = dataset.env
    observation = env.reset()

    ## observations for rendering
    rollout = [observation.copy()]

    total_reward = 0

    for t in range(args.max_episode_length):
        
        ## format current observation for conditioning
        conditions = {0: observation}
        action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)

        #env.render()
        import numpy as np
        # for i in range(len(samples.observations[0])):
        #     x = samples.observations[0][i]
        #     print("state:  ", np.unravel_index(np.argmax(x), env.shape), np.max(x))
        #     print("action:  ", np.argmax(samples.actions[0][i]))
        print(samples.observations, samples.actions, samples.rewards)
        print(np.max(samples.observations, axis=-1), np.argmax(samples.observations, axis=-1))
        print(np.max(samples.actions, axis=-1), np.argmax(samples.actions, axis=-1))

        # env.render()
        # dire = ['U', 'R', 'D', 'L']
        # print("action:   ", dire[np.argmax(action)], np.max(action))
        # for k, traj in enumerate(samples.observations):
        #     p = []
        #     o = np.zeros((4,12))
        #     for i in range(8):
        #         pos = np.argmax(traj[i, :])
        #         o[ np.unravel_index(pos, o.shape)] = i+1
        #         p.append( np.unravel_index(pos, o.shape))
        #     print(o)
        #     print("values:  ", samples.values[k])
        #     print(p)
        #     print([dire[i] for i in np.argmax(samples.actions[k], axis=-1)])
        # input()
        assert 0

        ## execute action in environment
        next_observation, reward, terminal, _ = env.step(action)

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
