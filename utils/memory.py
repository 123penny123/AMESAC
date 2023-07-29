import random
import torch
import numpy as np
from torch import FloatTensor
from collections import namedtuple
from collections import deque
from abc import ABC, abstractmethod
from typing import Optional,Union,Sequence
from gym.spaces import Space,Dict
from utils import EPS, discount_cumsum, space2shape


class OffPolicyBuffer():
    def __init__(self,                  
                 observation_space:Space,
                 action_space:Space,
                 buffer_size, 
                 batch_size,
                 seed, 
                 device,
                 task_one_hot=False,
                 num_tasks=1):
        random.seed(seed)
        self.obs_dim = observation_space.shape[0] + num_tasks * task_one_hot
        self.act_dim = action_space.shape[0]
        
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.num_tasks = num_tasks
        
        self.states = np.zeros((num_tasks, buffer_size, self.obs_dim))
        self.actions = np.zeros((num_tasks, buffer_size, self.act_dim))
        self.rewards = np.zeros((num_tasks, buffer_size,))
        self.next_states = np.zeros((num_tasks, buffer_size, self.obs_dim))
        self.dones = np.zeros((num_tasks, buffer_size,))
        # self.reps = np.zeros((num_tasks, buffer_size, self.obs_dim))
        
        self.ptr = np.zeros(num_tasks)
        self.size = np.zeros(num_tasks)
        
    def full(self):
        return self.size == self.buffer_size
    
    def clear(self):
        pass
        
    def store(self, state, action, reward, next_state, terminal, representation=None, idx=0):
        ptr = int(self.ptr[idx])
        self.states[idx][ptr] = state
        self.actions[idx][ptr] = action
        self.rewards[idx][ptr] = reward
        self.next_states[idx][ptr] = next_state
        self.dones[idx][ptr] = terminal
        
        self.ptr[idx] = (ptr+1)%self.buffer_size
        self.size[idx] = min(self.size[idx]+1, self.buffer_size)
    
    def sample(self, reshape=True):   # for mtmhsac, reshape=False
        b_s = np.zeros((self.num_tasks, self.batch_size, self.obs_dim))
        b_a = np.zeros((self.num_tasks, self.batch_size, self.act_dim))
        b_r = np.zeros((self.num_tasks, self.batch_size, ))
        b_s_ = np.zeros((self.num_tasks, self.batch_size, self.obs_dim))
        b_d = np.zeros((self.num_tasks, self.batch_size, ))
        b_rep = np.zeros((self.num_tasks, self.batch_size, self.obs_dim))
        
        for idx in range(self.num_tasks):          
            batch_choices = np.random.choice(int(self.size[idx]), self.batch_size)
            
            b_s[idx] = self.states[idx][batch_choices]
            b_a[idx] = self.actions[idx][batch_choices]
            b_r[idx] = self.rewards[idx][batch_choices]
            b_s_[idx] = self.next_states[idx][batch_choices]
            b_d[idx] = self.dones[idx][batch_choices]
            # b_rep[idx] = self.reps[idx][batch_choices]

        if reshape:
            b_s = b_s.reshape(self.num_tasks*self.batch_size, self.obs_dim)
            b_a = b_a.reshape(self.num_tasks*self.batch_size, self.act_dim)
            b_r = b_r.reshape(self.num_tasks*self.batch_size,)     
            b_s_ = b_s_.reshape(self.num_tasks*self.batch_size, self.obs_dim)
            b_d = b_d.reshape(self.num_tasks*self.batch_size, )
            b_rep = b_rep.reshape(self.num_tasks*self.batch_size, self.obs_dim)
        
        return b_s, b_a, b_r, b_s_, b_d, b_rep
    
def create_memory(shape: Optional[Union[tuple, dict]], nenvs: int, nsize: int):
    if shape == None:
        return None
    elif isinstance(shape, dict):
        memory = {}
        for key, value in zip(shape.keys(), shape.values()):
            if value is None:  # save an object type
                memory[key] = np.zeros([nenvs, nsize], dtype=object)
            else:
                memory[key] = np.zeros([nenvs, nsize] + list(value), dtype=np.float32)
        return memory
    elif isinstance(shape, tuple):
        return np.zeros([nenvs, nsize] + list(shape), np.float32)
    else:
        raise NotImplementedError


def store_element(data: Optional[Union[np.ndarray, dict, float]], memory: Union[dict, np.ndarray], ptr: int):
    if data is None:
        return
    elif isinstance(data, dict):
        for key, value in zip(data.keys(), data.values()):
            memory[key][:, ptr] = data[key]
    else:
        memory[:, ptr] = data


def sample_batch(memory: Optional[Union[np.ndarray, dict]], index: np.ndarray):
    if memory is None:
        return None
    elif isinstance(memory, dict):
        batch = {}
        for key, value in zip(memory.keys(), memory.values()):
            batch[key] = value[index]
        return batch
    else:
        return memory[index]



class DummyOnPolicyBuffer():
    def __init__(self,
                 observation_space: Space,
                 action_space: Space,
                 auxiliary_shape: Optional[dict],
                 nenvs: int,
                 nsize: int,
                 nminibatch: int,
                 gamma: float = 0.99,
                 lam: float = 0.95):
        super(DummyOnPolicyBuffer, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.auxiliary_shape = auxiliary_shape
        self.size, self.ptr = 0, 0
        
        self.nenvs, self.nsize, self.nminibatch = nenvs, nsize, nminibatch
        self.gamma, self.lam = gamma, lam
        self.start_ids = np.zeros(self.nenvs, np.int64)
        self.observations = create_memory(space2shape(self.observation_space), self.nenvs, self.nsize)
        self.actions = create_memory(space2shape(self.action_space), self.nenvs, self.nsize)
        self.rewards = create_memory((), self.nenvs, self.nsize)
        self.returns = create_memory((), self.nenvs, self.nsize)
        self.terminals = create_memory((), self.nenvs, self.nsize)
        self.advantages = create_memory((), self.nenvs, self.nsize)
        self.auxiliary_infos = create_memory(self.auxiliary_shape, self.nenvs, self.nsize)

    @property
    def full(self):
        return self.size >= self.nsize

    def clear(self):
        self.ptr, self.size = 0, 0
        self.observations = create_memory(space2shape(self.observation_space), self.nenvs, self.nsize)
        self.actions = create_memory(space2shape(self.action_space), self.nenvs, self.nsize)
        self.rewards = create_memory((), self.nenvs, self.nsize)
        self.returns = create_memory((), self.nenvs, self.nsize)
        self.advantages = create_memory((), self.nenvs, self.nsize)
        self.auxiliary_infos = create_memory(self.auxiliary_shape, self.nenvs, self.nsize)

    def store(self, obs, acts, rews, rets, terminals, aux_info):
        store_element(obs, self.observations, self.ptr)
        store_element(acts, self.actions, self.ptr)
        store_element(rews, self.rewards, self.ptr)
        store_element(rets, self.returns, self.ptr)
        store_element(terminals, self.terminals, self.ptr)
        store_element(aux_info, self.auxiliary_infos, self.ptr)
        self.ptr = (self.ptr + 1) % self.nsize
        self.size = min(self.size + 1, self.nsize)

    def finish_path(self, val, i):
        if self.full:
            path_slice = np.arange(self.start_ids[i], self.nsize).astype(np.int32)
        else:
            path_slice = np.arange(self.start_ids[i], self.ptr).astype(np.int32)
        rewards = np.append(np.array(self.rewards[i, path_slice]), [val], axis=0)
        critics = np.append(np.array(self.returns[i, path_slice]), [val], axis=0)
        returns = discount_cumsum(rewards, self.gamma)[:-1]
        deltas = rewards[:-1] + self.gamma * critics[1:] - critics[:-1]
        advantages = discount_cumsum(deltas, self.gamma * self.lam)
        self.returns[i, path_slice] = returns
        self.advantages[i, path_slice] = advantages
        self.start_ids[i] = self.ptr

    def sample(self):
        assert self.full, "Not enough transitions for on-policy buffer to random sample"

        env_choices = np.random.choice(self.nenvs, self.nenvs * self.nsize // self.nminibatch)
        step_choices = np.random.choice(self.nsize, self.nenvs * self.nsize // self.nminibatch)

        obs_batch = sample_batch(self.observations, tuple([env_choices, step_choices]))
        act_batch = sample_batch(self.actions, tuple([env_choices, step_choices]))
        ret_batch = sample_batch(self.returns, tuple([env_choices, step_choices]))
        adv_batch = sample_batch(self.advantages, tuple([env_choices, step_choices]))
        aux_batch = sample_batch(self.auxiliary_infos, tuple([env_choices, step_choices]))
        adv_batch = (adv_batch - np.mean(self.advantages)) / (np.std(self.advantages) + EPS)

        return obs_batch, act_batch, ret_batch, adv_batch, aux_batch