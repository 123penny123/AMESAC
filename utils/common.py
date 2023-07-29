import os
import yaml
import numpy as np
import scipy.signal
from gym.spaces import Space,Dict,Box,Discrete
from argparse import Namespace
from typing import Sequence,Union


EPS=1e-8
def get_config(yaml_path):
    config = yaml.load(open(yaml_path).read(),Loader=yaml.Loader)
    config = Namespace(**config)
    return config

def create_directory(path):
    dir_split = path.split("/")
    current_dir = dir_split[0] + "/"
    for i in range(1,len(dir_split)):
        if not os.path.exists(current_dir):
            os.mkdir(current_dir)
        current_dir = current_dir + dir_split[i] + "/"
        
def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumsum(x,discount=0.99):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def space2shape(observation_space:Space):
    if isinstance(observation_space,Dict):
        return {key:observation_space[key].shape for key in observation_space.spaces.keys()}
    else:
        return observation_space.shape

def dict_reshape(keys,dict_list:Sequence[dict]):
    results = {}
    for key in keys():
        results[key] = np.array([element[key] for element in dict_list],np.float32)
    return results

