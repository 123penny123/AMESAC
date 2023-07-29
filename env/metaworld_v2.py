import metaworld
import gym
import random
import numpy as np
from gym.spaces import Dict,Discrete,Box

class MT1_Env(gym.Env):
    def __init__(self, config):
        mt1 = metaworld.MT1(config.env_name) # Construct the benchmark, sampling tasks

        self.env = mt1.train_classes[config.env_name]()  # Create an environment with task `pick_place`
        task = random.choice(mt1.train_tasks)
        self.env.set_task(task)  # Set task
        
        self.env._freeze_rand_vec = False
        
        self.config = config
        self.timelimit = config.timelimit
        
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        super(MT1_Env,self).__init__()    
    
    def reset(self):
        self.timesteps = 0
        obs = self.env.reset()
        return obs
        
    def step(self, action):
        # revise the correct action range
        obs, reward, done, info = self.env.step(action)
        # increase the timesteps
        self.timesteps += 1
        if self.timesteps >= self.config.timelimit:
            done = True
        return obs, reward, done, info
        
    def render(self):
        self.env.render()
    
    def seed(self,seed:int):
        self.env.seed(seed)
    
    def close(self):
        self.env.close()

class MT10_Env(gym.Env):
    def __init__(self, task_one_hot, config):
        self.config = config
        self.task_one_hot = task_one_hot
        self.timelimit = config.timelimit
        self.num_tasks = config.num_tasks
        self.timesteps = np.zeros((self.num_tasks, ))
        
        mt10 = metaworld.MT10()
        self.train_classes = mt10.train_classes
        self.train_tasks = mt10.train_tasks
        self.envs = []
        self._set_env()
        
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
    def _set_env(self):
        num = 0
        for name, env_cls in self.train_classes.items():
            num += 1
            env = env_cls()
            task = random.choice([task for task in self.train_tasks
                                    if task.env_name == name])
            env.set_task(task)
            env._freeze_rand_vec=False
            self.envs.append(env)
            if num == self.num_tasks:
                break
            
    def _taskidx_onehot(self, idx):
        one_hot = np.zeros((self.num_tasks,))
        one_hot[idx] = 1.
        return one_hot
                
    def reset(self, idx):
        self.seed(np.random.randint(0, 10000))
        self.timesteps[idx] = 0

        obs = self.envs[idx].reset()
        if self.task_one_hot:
            idx = self._taskidx_onehot(idx)
            obs = np.concatenate((obs, idx))
        return obs
    
    def step(self, action, idx):
        obs, reward, done, info = self.envs[idx].step(action)
        # increase the timesteps
        self.timesteps[idx] += 1
        if self.timesteps[idx] >= self.config.timelimit:
            done = True
        if self.task_one_hot:
            idx = self._taskidx_onehot(idx)
            obs = np.concatenate((obs, idx))
        
        return obs, reward, done, info
    
    def render(self, idx):
        return self.envs[idx].render('rgb_array')
    
    def seed(self, seed):
        for idx in range(self.num_tasks):
            self.envs[idx].seed(seed) 
            
    def close(self, idx):
        self.envs[idx].close()
        
class MT50_Env(gym.Env):
    def __init__(self, task_one_hot, config):
        self.config = config
        self.task_one_hot = task_one_hot
        self.timelimit = config.timelimit
        self.num_tasks = config.num_tasks
        self.timesteps = np.zeros((self.num_tasks, ))
        
        mt50 = metaworld.MT50()
        self.train_classes = mt50.train_classes
        self.train_tasks = mt50.train_tasks
        self.envs = []
        self._set_env()
        
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
    def _set_env(self):
        num = 0
        for name, env_cls in self.train_classes.items():
            num += 1
            env = env_cls()
            task = random.choice([task for task in self.train_tasks
                                    if task.env_name == name])
            env.set_task(task)
            env._freeze_rand_vec=False
            self.envs.append(env)
            if num == self.num_tasks:
                break
            
    def _taskidx_onehot(self, idx):
        one_hot = np.zeros((self.num_tasks,))
        one_hot[idx] = 1.
        return one_hot
                
    def reset(self, idx):
        self.seed(np.random.randint(0, 10000))
        self.timesteps[idx] = 0

        obs = self.envs[idx].reset()
        if self.task_one_hot:
            idx = self._taskidx_onehot(idx)
            obs = np.concatenate((obs, idx))
        return obs
    
    def step(self, action, idx):
        # revise the correct action range
        obs, reward, done, info = self.envs[idx].step(action)
        # increase the timesteps
        self.timesteps[idx] += 1
        if self.timesteps[idx] >= self.config.timelimit:
            done = True
        if self.task_one_hot:
            idx = self._taskidx_onehot(idx)
            obs = np.concatenate((obs, idx))
        
        return obs, reward, done, info
    
    def render(self, idx):
        self.envs[idx].render()
    
    def seed(self, seed):
        for idx in range(self.num_tasks):
            self.envs[idx].seed(seed) 
            
    def close(self, idx):
        self.envs[idx].close()

