import gym
from .metaworld_v2 import MT1_Env#, MT10_Env, MT50_Env
from argparse import Namespace
def make_env_func(task:int,config:Namespace):
    def _thunk():
        if task == 0:
            env = MT1_Env(config)
        # elif task == 1:
        #     env = MT10_Env(config)
        # elif task == 3:
        #     env = MT50_Env(config)
        elif task == 4:
            env = gym.make("HalfCheetah-v2") 
        elif task == 5:
            env = gym.make("Walker2d-v3") 
        elif task == 6:
            raise NotImplementedError
        return env
    return _thunk
        