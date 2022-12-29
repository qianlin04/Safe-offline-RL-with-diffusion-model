import pdb

import diffuser.sampling as sampling
import diffuser.utils as utils
from cost import *
import wandb
import numpy as np
import os

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'walker2d-medium-replay-v2'
    config: str = 'config.locomotion'
    project: str = 'SafeRLDiffusion'
    cost_threshold: float = None
    ratio_of_maxthreshold: float = None
    n_test_episode: int = 10
    use_wandb: int = True
    gpu_id: str = '0'

args = Parser().parse_args('plan')
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

print(args.dataset, args.ratio_of_maxthreshold)
if args.cost_threshold is None:
    if args.ratio_of_maxthreshold is None:
        args.cost_threshold = np.inf
    else:
        if args.test_cost_with_discount:
            args.cost_threshold = MAX_COST_DISCOUNT_THRESHOLD[args.dataset] * args.ratio_of_maxthreshold
        else:
            args.cost_threshold = MAX_COST_THRESHOLD[args.dataset] * args.ratio_of_maxthreshold

init_cost_threshold = args.cost_threshold # 1e6 #82.75 #1.5
if not args.test_cost_with_discount:
    if args.test_cost_with_fixed_length:
        init_cost_threshold = init_cost_threshold * (1-args.discount**args.max_episode_length) \
            / (1-args.discount) / args.max_episode_length
    else:
        init_cost_threshold = init_cost_threshold * THRESHOLD_RATIO[args.dataset] 
args.init_cost_threshold = init_cost_threshold

def cost_func(*a, **kwargs):
    #kwargs['is_single_step'] = True
    #kwargs['binarization_threshold'] = 2.5
    kwargs['env_name'] = args.dataset.split("-")[0]
    return vel_cost(*a, **kwargs)


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
all_return = []
all_episode_len = []
all_total_cost = []
all_discount_total_cost = []

for n_test_episode in range(args.n_test_episode):

    args.n_test_episode = n_test_episode
    if args.use_wandb:
        wandb.init(
            project=args.project,
            config=args,
            name=f'{args.dataset}_{args.seed}_{args.cost_threshold:.2f}',
            group='default',
        )

    history = []
    remain_cost = init_cost_threshold 
    n_single_step_break = 0
    total_reward = 0
    total_cost = 0
    discount_total_cost = 0

    env = dataset.env
    observation = env.reset()
    if not "Safe" in args.dataset:
        cost = eval_cost_from_env(env, history)

    ## observations for rendering
    rollout = [observation.copy()]

    for t in range(args.max_episode_length):

        if t % 10 == 0: print(args.savepath, flush=True)

        ## save state for rendering only
        # state = env.state_vector().copy()

        ## format current observation for conditioning
        cost_threshold = remain_cost / (args.discount**(t)) #* (1-args.discount**(args.max_episode_length-t)))

        conditions = {0: observation}
        action, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose,
                                 cost_threshold=cost_threshold, plan_horizon=args.plan_horizon)

        ## execute action in environment
        next_observation, reward, terminal, info = env.step(action)
        if 'cost' in info:
            cost = info['cost']
        else:
            cost = eval_cost_from_env(env, history)

        ## print reward and score
        n_single_step_break += (cost > single_check_threshold)
        total_reward += reward
        discount_total_cost += cost * (args.discount ** t)
        if args.test_cost_with_discount:
            total_cost += cost * (args.discount ** t)
        else:
            total_cost += cost
        remain_cost = init_cost_threshold - discount_total_cost

        score = env.get_normalized_score(total_reward) if hasattr(env, 'get_normalized_score') else 0
        
        if args.use_wandb:
            wandb.log({"score": score, 'time_step':t}) 
            wandb.log({"return": total_reward, 'time_step':t})
            wandb.log({"values": samples.values[0], 'time_step':t}) 
            wandb.log({"normed_total_cost": total_cost/args.cost_threshold, 'time_step':t})
            wandb.log({"total_cost": total_cost, 'time_step':t}) 
            wandb.log({"normed_discount_total_cost": discount_total_cost/init_cost_threshold, 'time_step':t})
            wandb.log({"discount_total_cost": discount_total_cost, 'time_step':t}) 
            wandb.log({"cost_threshold": cost_threshold, 'time_step':t}) 
            wandb.log({"cost_values": samples.costs[0], 'time_step':t}) 

        print(
            f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | values: {samples.values[0]} | scale: {args.scale} | terminal: {terminal} \n'
            f'cost: {cost:.2f} | normed_total_cost : {total_cost/args.cost_threshold:.2f} | total_cost : {total_cost:.2f} | '
            f'normed_discount_total_cost : {discount_total_cost/init_cost_threshold:.2f} | discount_total_cost : {discount_total_cost:.2f} | '
            f'cost_threshold : {cost_threshold:.2f} | cost_values: {samples.costs[0]} | n_single_break: {n_single_step_break} | '
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

    print()
    all_results.append(f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | values: {samples.values[0]} | scale: {args.scale} | terminal: {terminal} \n'
            f'cost: {cost:.2f} | normed_total_cost : {total_cost/args.cost_threshold:.2f} | total_cost : {total_cost:.2f} | '
            f'normed_discount_total_cost : {discount_total_cost/init_cost_threshold:.2f} | discount_total_cost : {discount_total_cost:.2f} | '
            f'cost_threshold : {cost_threshold:.2f} | cost_values: {samples.costs[0]} | n_single_break: {n_single_step_break} | '
            f'cost_func: {("single_" if is_single_step_constraint else "")+("bin_" if binarization_threshold is not None else "")+cost_func_name}'
            )
    all_scores.append(score)
    all_total_cost.append(total_cost)
    all_discount_total_cost.append(discount_total_cost)
    all_episode_len.append(t+1)
    all_return.append(total_reward)
    all_normed_total_cost = np.array(all_total_cost)/args.cost_threshold
    all_normed_discount_total_cost = np.array(all_discount_total_cost)/init_cost_threshold

    for res in all_results:
        print(res)
    print("all scores:  ", all_scores)
    print("all normed total cost: ", all_normed_total_cost)
    print("all total cost: ", all_total_cost)
    print("all normed discount total cost: ", all_normed_discount_total_cost)
    print("all total discount cost: ", all_discount_total_cost)
    print("all return:  ", all_return)
    print("all episode len:  ", all_episode_len)
    import numpy as np
    print("mean scores:      ", np.mean(all_scores), np.std(all_scores))
    print("mean normed total cost:      ", np.mean(all_normed_total_cost), np.std(all_normed_total_cost))
    print("mean total cost:      ", np.mean(all_total_cost), np.std(all_total_cost))
    print("mean normed discount total cost:      ", np.mean(all_normed_discount_total_cost), np.std(all_normed_discount_total_cost))
    print("mean total discount cost:      ", np.mean(all_discount_total_cost), np.std(all_discount_total_cost))
    print("mean return:      ", np.mean(all_return), np.std(all_return))
    print("mean episode len:      ", np.mean(all_episode_len), np.std(all_episode_len))

    if args.use_wandb:
        wandb.log({'final/score': score, 'n_test_episode': n_test_episode})
        wandb.log({"final/normed_total_cost": total_cost/args.cost_threshold, 'n_test_episode': n_test_episode}) 
        wandb.log({"final/total_cost": total_cost, 'n_test_episode': n_test_episode}) 
        wandb.log({"final/normed_discount_total_cost": discount_total_cost/init_cost_threshold, 'n_test_episode': n_test_episode}) 
        wandb.log({"final/discount_total_cost": discount_total_cost, 'n_test_episode': n_test_episode}) 
        wandb.log({"final/episode_len": t+1, 'n_test_episode': n_test_episode}) 
        wandb.log({"final/return": total_reward, 'n_test_episode': n_test_episode})

    if args.use_wandb:
        wandb.finish()
