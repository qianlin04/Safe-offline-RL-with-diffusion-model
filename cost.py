import numpy as np

MAX_COST_THRESHOLD = {
    'hopper-medium-v2': 129.940,
    'walker2d-medium-v2': 159.512,
    'halfcheetah-medium-v2': 414.008,
    'hopper-medium-replay-v2': 139.884,
    'walker2d-medium-replay-v2': 160.251,
    'halfcheetah-medium-replay-v2': 384.375,
    'hopper-medium-expert-v2': 168.918,
    'walker2d-medium-expert-v2': 234.076,
    'halfcheetah-medium-expert-v2': 730.746,
}

def eval_cost(history, cost_func_name="vel_cost", is_single_step=False, binarization_threshold=None):
    observation, action, next_observation, reward, terminal, info = history[-1]
    if cost_func_name=="vel_cost":
        cost = 0 if not 'x_velocity' in info else np.abs(info['x_velocity'])
    elif cost_func_name=="action_cost":
        cost = np.sum(np.square(action))
    if binarization_threshold is not None:
        cost = (cost > binarization_threshold)
    # if not is_single_step:
    #     cost_threshold = (remain_cost - cost * (discount ** t))
    return cost

def eval_cost_from_env(env, history, cost_func_name="vel_cost"):
    if cost_func_name=="vel_cost":
        if len(history)==0:
            history.append(env.sim.data.qpos[0])
            return 0
        x_position_before = history[-1]
        history.append(env.sim.data.qpos[0])
        x_position_after = history[-1]
        x_velocity = (x_position_after - x_position_before) / env.dt
        return np.abs(x_velocity)
    else:
        assert 0

def action_cost(start, field, is_single_step=False, env_name = "hopper", binarization_threshold=None): #constraint with abs(action)
    actions = field['actions'][start:]
    # cost = np.mean(np.abs(actions), axis=1, keepdims=True)
    cost = np.sum(np.square(actions), axis=1, keepdims=True)
    if binarization_threshold is not None: 
        cost = cost > binarization_threshold
    #print(np.sum(np.square(actions), axis=1) > binarization_threshold)
    if is_single_step: 
        cost[1:] = 0
    return cost

def qvel_cost(start, field, infos_qvel, is_single_step=False, env_name = "hopper"):
    infos_qvel = field['infos/qvel'][start:]
    cost = np.mean(np.abs(infos_qvel), axis=1, keepdims=True)
    if is_single_step: cost[1:] = 0
    return cost

def qpos_cost(start, field, is_single_step=False, env_name = "hopper"):
    infos_qpos = field['infos/qpos'][start:]
    cost = np.mean(np.abs(infos_qpos), axis=1, keepdims=True)
    if is_single_step: cost[1:] = 0
    return cost

def vel_cost(start, field, is_single_step=False, env_name = "hopper", binarization_threshold=None):
    dt = {"hopper": 0.008, 
          "walker2d": 0.008, 
          "halfcheetah": 0.05}
    
    infos_qpos, path_length = field['infos/qpos'], field['path_lengths']
    vel_delta = np.zeros_like(infos_qpos[start:, 0])
    vel_delta[1:path_length-start] = infos_qpos[start+1:path_length, 0] - infos_qpos[start:path_length-1, 0]
    if start != 0:
        vel_delta[0] = infos_qpos[start, 0] - infos_qpos[start-1, 0]
    x_velocity = vel_delta / dt[env_name]
    if binarization_threshold is not None: 
        x_velocity = x_velocity > binarization_threshold
    if is_single_step: 
        x_velocity[1:] = 0
    cost = np.abs(x_velocity)[:, np.newaxis]
    return cost

# def pendulum_cost(start, field, is_single_step=False, env_name = "hopper", binarization_threshold=None):
#     states, actions, next_states = field['states'][start:, :], field['actions'][start:, :], field['next_states'][start:, :]


def healthy_cost(start, field, is_single_step=False, env_name = "hopper"):
    healthy_param = {
        "hopper" :{
            "healthy_state_range": (-100.0, 100.0), 
            "healthy_z_range": (0.7, float('inf')),
            "healthy_angle_range": (-0.2, 0.2),
        },
        "walker2d" :{
            "healthy_state_range": None, 
            "healthy_z_range": (0.8, 2.0),
            "healthy_angle_range": (-1.0, 1.0),
        },
        "halfcheetah" :{
            "healthy_state_range": None, 
            "healthy_z_range": None,
            "healthy_angle_range": None,
        },
    }

    infos_qpos, path_length, obs = field['infos/qpos'], field['path_lengths'], field['observations']
    healthy_state_range = healthy_param[env_name]['healthy_state_range']
    healthy_z_range = healthy_param[env_name]['healthy_z_range']
    healthy_angle_range = healthy_param[env_name]['healthy_angle_range']
    
    z, angle = infos_qpos[start:, 1], infos_qpos[start:, 2]
    state = obs[start:]

    if healthy_state_range is not None:
        min_state, max_state = healthy_state_range
        healthy_state = np.all(
            np.logical_and(min_state < state, state < max_state), axis=1)
    else:
        healthy_state = True
    
    if healthy_z_range is not None:
        min_z, max_z = healthy_z_range
        healthy_z = np.logical_and(min_z < z, z < max_z)
    else:
        healthy_z = True
    
    if healthy_angle_range is not None:
        min_angle, max_angle = healthy_angle_range
        healthy_angle = np.logical_and(min_angle < angle, angle < max_angle) 
    else:
        healthy_angle_range = True

    print(healthy_angle)
    is_healthy = healthy_state * healthy_z * healthy_angle
    is_healthy[path_length-start:] = False
    is_healthy = is_healthy[:, np.newaxis]
    return is_healthy

def get_threshold(env, constraint='velocity'):

    if constraint == 'circle':
        return 50
    else:
        thresholds = {'Ant-v3': 103.115,
                      'HalfCheetah-v3': 151.989,
                      'Hopper-v3': 82.748,
                      'Humanoid-v3': 20.140,
                      'Swimmer-v3': 24.516,
                      'Walker2d-v3': 81.886
                      }
        return thresholds[env]


'''
if constraint == 'velocity':
                    if 'y_velocity' not in info:
                        cost = np.abs(info['x_velocity'])
                    else:
                        cost = np.sqrt(info['x_velocity'] ** 2 + info['y_velocity'] ** 2)
                elif constraint == 'circle':
                    cost = info['cost']
                else:
                    cost = info.get('cost', 0)
'''