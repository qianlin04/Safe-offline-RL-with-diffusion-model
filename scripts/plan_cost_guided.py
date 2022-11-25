import pdb

import diffuser.sampling as sampling
import diffuser.utils as utils
from cost import *

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'walker2d-medium-replay-v2'
    config: str = 'config.locomotion'
    cost_threshold: float = 1e6
    n_test_episode: int = 5

args = Parser().parse_args('plan')

def cost_func(*a, **kwargs):
    #kwargs['is_single_step'] = True
    #kwargs['binarization_threshold'] = 2.5
    kwargs['env_name'] = args.dataset.split("-")[0]
    return vel_cost(*a, **kwargs)

init_cost_threshold = args.cost_threshold # 1e6 #82.75 #1.5
is_single_step_constraint = False
single_check_threshold = 2.0
binarization_threshold = None #2.5
cost_func_name = "vel_cost"

if is_single_step_constraint:
    name = "single_step_" + cost_func_name
elif binarization_threshold:
    name = "bin_" + cost_func_name
else:
    name = cost_func_name
args.cost_value_loadpath = f'{name}_values/defaults_H{args.horizon}_T{args.n_diffusion_steps}_d{args.discount}'

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
cost_value_experiment = load_diffusion_func(
    args.loadbase, args.dataset, args.cost_value_loadpath,
    epoch=args.cost_value_epoch, seed=args.seed,
)

## ensure that the diffusion model and value function are compatible with each other
utils.check_compatibility(diffusion_experiment, value_experiment)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

## initialize value guide
value_function = value_experiment.ema
cost_function = cost_value_experiment.ema
guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()
cost_guide_config = utils.Config(args.guide, model=cost_function, verbose=False)
cost_guide = cost_guide_config()

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
    sample_fn=sampling.n_step_guided_and_constrained_p_sample,
    cost_guide = cost_guide,
    cost_threshold = 0,
    cost_grad_weight=args.cost_grad_weight,
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
all_total_cost = [] 

for _ in range(args.n_test_episode):

    env = dataset.env
    observation = env.reset()

    ## observations for rendering
    rollout = [observation.copy()]
    history = []
    remain_cost = init_cost_threshold 
    n_single_step_break = 0

    total_reward = 0
    total_cost = 0
    for t in range(args.max_episode_length):

        if t % 10 == 0: print(args.savepath, flush=True)

        ## save state for rendering only
        state = env.state_vector().copy()

        ## format current observation for conditioning
        if args.test_cost_with_discount:
            cost_threshold = remain_cost / (args.discount**(t) * (1-args.discount**(args.max_episode_length-t)))
        else:
            cost_threshold = remain_cost * (1-args.discount**args.max_episode_length) / ((args.max_episode_length-t) * (1-args.discount))
        conditions = {0: observation}
        action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose, cost_threshold=cost_threshold)

        ## execute action in environment
        next_observation, reward, terminal, info = env.step(action)
        history.append((observation, action, next_observation, reward, terminal, info))
        cost = eval_cost(history, cost_func_name=cost_func_name, binarization_threshold=binarization_threshold, is_single_step=is_single_step_constraint)

        ## print reward and score
        n_single_step_break += (cost > single_check_threshold)
        total_reward += reward
        if args.test_cost_with_discount:
            total_cost += cost * (args.discount ** t)
        else:
            total_cost += cost
        remain_cost = init_cost_threshold-total_cost

        score = env.get_normalized_score(total_reward)
        
        print(
            f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | values: {samples.values[0]} | scale: {args.scale} | terminal: {terminal} \n'
            f'cost: {cost:.2f} | total_cost : {total_cost:.2f} | cost_threshold : {cost_threshold:.2f} | cost_values: {samples.costs[0]} | n_single_break: {n_single_step_break} | '
            f'cost_func: {("single_" if is_single_step_constraint else "")+("bin_" if binarization_threshold is not None else "")+cost_func_name}',
            flush=True,
        )

        ## update rollout observations
        rollout.append(next_observation.copy())

        ## render every `args.vis_freq` steps
        #logger.log(t, samples, state, rollout)

        if terminal:
            break

        observation = next_observation

    ## write results to json file at `args.savepath`
    #logger.finish(t, score, total_reward, terminal, diffusion_experiment, value_experiment)

    all_results.append(f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | values: {samples.values[0]} | scale: {args.scale} | '
            f'cost: {cost:.2f} | total_cost : {total_cost:.2f} | cost_threshold : {cost_threshold:.2f} | cost_values: {samples.costs[0]} | n_single_break: {n_single_step_break} | '
            f'cost_func: {("single_" if is_single_step_constraint else "")+("bin_" if binarization_threshold is not None else "")+cost_func_name}')
    all_scores.append(score)
    all_total_cost.append(total_cost)

    for res in all_results:
        print(res)
    print("score: ", all_scores)
    print(np.mean(all_scores))
    print("total_cost: ", all_total_cost)
    print(np.mean(all_total_cost))
