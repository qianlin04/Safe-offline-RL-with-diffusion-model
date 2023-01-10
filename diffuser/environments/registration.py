import gym
import copy

ENVIRONMENT_SPECS = (
    {
        'id': 'HopperFullObs-v2',
        'entry_point': ('diffuser.environments.hopper:HopperFullObsEnv'),
    },
    {
        'id': 'HalfCheetahFullObs-v2',
        'entry_point': ('diffuser.environments.half_cheetah:HalfCheetahFullObsEnv'),
    },
    {
        'id': 'Walker2dFullObs-v2',
        'entry_point': ('diffuser.environments.walker2d:Walker2dFullObsEnv'),
    },
    {
        'id': 'AntFullObs-v2',
        'entry_point': ('diffuser.environments.ant:AntFullObsEnv'),
    },
    
)

TOY_ENVIRONMENT_SPECS = (
    {
        'id': 'TwoStepMDP-v0',
        'entry_point': ('diffuser.environments.two_step_mdp:TwoStepMDP'),
        'max_episode_steps' :200,
        'kwargs': {},
    },
    {
        'id': 'mycliffwalking-v0',
        'entry_point': ('diffuser.environments.mycliffwalking:CliffWalkingEnv'),
        'kwargs': {},
        'max_episode_steps':200,
    },
    {
        'id': 'myroulette-v0',
        'entry_point': ('diffuser.environments.myroulette:RouletteEnv'),
        'kwargs': {},
        'max_episode_steps':100,
    },
    {
        'id' : 'myFrozenLake-v0',
        'entry_point': ('diffuser.environments.myfrozen_lake:FrozenLakeEnv'),
        'kwargs':{'map_name' : '4x4'},
        'max_episode_steps' :100,
        'reward_threshold': 0.78,
    },
    {
        'id' : 'myFrozenLake8x8-v0',
        'entry_point': ('diffuser.environments.myfrozen_lake:FrozenLakeEnv'),
        'kwargs':{'map_name' : '8x8'},
        'max_episode_steps' :200,
        'reward_threshold': 0.99,
    },
    {
        'id': 'SafePendulum-v0',
        'entry_point': ('diffuser.environments.single_pendulum:SafePendulumEnv'),
        'kwargs': {},
        'max_episode_steps':200,
    },
    {
        'id': 'SafeDoublePendulum-v0',
        'entry_point': ('diffuser.environments.double_pendulum:SafeDoublePendulumEnv'),
        'kwargs': {},
        'max_episode_steps':200,
    },
    {
        'id': 'SafeReacher-v0',
        'entry_point': ('diffuser.environments.reacher:SafeReacherEnv'),
        'kwargs': {},
        'max_episode_steps':50,
    },
    {
        'id': 'ocpm-v0',
        'entry_point': ('diffuser.environments.ocpm:OCPMEnv'),
        'max_episode_steps':200,
    },
)

def register_environments():
    try:
        for environment in ENVIRONMENT_SPECS:
            gym.register(**environment)

        gym_ids = tuple(
            environment_spec['id']
            for environment_spec in  ENVIRONMENT_SPECS)

        gym_ids = gym_ids + tuple(
            environment_spec['id']
            for environment_spec in  TOY_ENVIRONMENT_SPECS)

        for environment in TOY_ENVIRONMENT_SPECS:
            gym.register(**environment)
            env_id, ver_id = environment['id'].split('-')
            for datatype in ['random', 'medium-replay', 'medium', 'mix']:
                env = copy.deepcopy(environment)
                env['id'] = '-'.join([env_id, datatype, ver_id])
                env['kwargs']['dataset_name'] = datatype
                gym.register(**env)

        return gym_ids
    except:
        print('[ diffuser/environments/registration ] WARNING: not registering diffuser environments')
        return tuple()