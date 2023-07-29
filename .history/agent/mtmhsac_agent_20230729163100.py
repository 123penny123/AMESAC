import torch
import time
import cv2
import numpy as np
import pickle as pkl
from tqdm import tqdm

from learner import MTMHSAC_Learner
from utils import OffPolicyBuffer,create_directory
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter


class MTMHSAC_Agent():
    def __init__(self,
                 env_config,
                 agent_config,
                 envs,
                 eval_envs,
                 policy,
                 max_setps=1000):
        self.env_config = env_config
        self.agent_config = agent_config
        self.envs = envs
        self.eval_envs = eval_envs
        self.policy = policy
        
        self.num_tasks = env_config.num_tasks
        self.buffer_size = agent_config.buffer_size
        self.batch_size = agent_config.batch_size
        self.seed = agent_config.seed
        self.max_steps = max_setps
        
        self.num_steps = agent_config.num_steps
        self.start_steps = agent_config.start_steps
        self.update_every = agent_config.update_every
        self.reshape = agent_config.memory_sample_reshape
        
        self.observation_space = envs.observation_space
        self.action_space = envs.action_space
        
        self.writer = SummaryWriter(agent_config.logdir)
        self.memory = OffPolicyBuffer(self.observation_space,
                                        self.action_space,
                                        self.buffer_size,
                                        self.batch_size,
                                        self.seed,
                                        agent_config.device,
                                        agent_config.task_one_hot,
                                        env_config.num_tasks)
        self.learner = MTMHSAC_Learner(self.action_space,
                                        policy,
                                        self.writer,
                                        agent_config.device,
                                        env_config.num_tasks,
                                        agent_config.modeldir,
                                        agent_config.policy_type,
                                        agent_config.automatic_entropy_tuning,
                                        agent_config.loss_reweight,
                                        agent_config.tau,
                                        agent_config.alpha,
                                        agent_config.gamma,
                                        agent_config.lr)
        create_directory(self.agent_config.modeldir)
        create_directory(self.agent_config.logdir)
        super(MTMHSAC_Agent,self).__init__()
    
    def _process_observation(self,observations):
        pass
        
    def _process_reward(self,rewards):
        pass
    
    def _save_model(self, steps):
        path = "{}/{}.pkl".format(self.agent_config.modeldir, int(steps/10000))
        torch.save(self.policy.state_dict(), path)
        
    def _load_model(self, path):
        self.policy.load_state_dict(torch.load(path,map_location=torch.device('cuda:0')))
        self.policy.eval()
    
    def _action(self, state, deterministic=False, plot=False):
        return self.policy.act(state, deterministic, plot)
    
    def _render(self, idx, steps):
        if not self.env_config.name == 'mujoco-mt':
            self.eval_envs.render(idx)
        else:
            img = self.envs.render(idx)
            img = cv2.resize(img, (150,150), interpolation=cv2.INTER_CUBIC).copy()
            text = "%.1fsteps"%steps
            img_text = cv2.putText(img, text, (5,50 ), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            self.concat_image[:,idx*150:idx*150+150,:] = img_text
            cv2.imshow('mujoco',self.concat_image)
            cv2.waitKey(1) 
          
    def train(self):        
        episodes = np.zeros((self.num_tasks,))
        scores = np.zeros((self.num_tasks,))
        returns = np.zeros((self.num_tasks,))
        success = np.zeros((self.num_tasks,))
        done_num = 0 
        concat_image = np.zeros((150,900,3),np.uint8)
        
        if self.agent_config.task_one_hot:
            obs = np.zeros((self.num_tasks,self.observation_space.shape[0]+self.num_tasks))
            obs_ = np.zeros((self.num_tasks,self.observation_space.shape[0]+self.num_tasks))
        else:
            obs = np.zeros((self.num_tasks,self.observation_space.shape[0]))
            obs_ = np.zeros((self.num_tasks,self.observation_space.shape[0]))
    
        for idx in range(self.num_tasks):
            obs[idx] = self.envs.reset(idx)  # Reset environment
            
        for steps in tqdm(range(self.num_steps)):
            
            # self.rewards_total = []
            for idx in range(self.num_tasks):
                if steps < self.start_steps:
                    act = np.random.uniform(-1, 1, (self.action_space.shape[0], ))
                else:
                    act = self._action(obs[idx]).squeeze(0)   # Sample action from policy
                    
                next_obs,reward,done,info = self.envs.step(act, idx)    
                
                scores[idx] += reward
                returns[idx] = returns[idx]*self.agent_config.gamma + reward
                
                masked_done = False if self.envs.timesteps[idx] >= self.env_config.timelimit else done
                self.memory.store(obs[idx],act,reward,next_obs,masked_done,idx=idx)
                obs_[idx] = next_obs
                
                if done:
                    done_num += 1
                    success[idx] += info['success']
                    
                    obs_[idx] = self.envs.reset(idx)
                    self.writer.add_scalars("train-scores-step-%d"%(idx//10),{"env-%d"%idx:scores[idx]},steps)
                    episodes[idx] += 1
                    scores[idx] = 0
                    returns[idx] = 0
                
                    if episodes[-1] % 10 == 0 and done_num % 10 == 0:  # store every 100 tests
                        self.writer.add_scalar("train-success-step",sum(success),steps)
                        success = np.zeros((self.num_tasks,)) 
                        done_num = 0
            
            if steps > self.start_steps and steps % self.update_every == 0:
                for _ in range(self.update_every):
                    b_s, b_a, b_r, b_s_, b_d, b_rep = self.memory.sample(reshape=self.reshape)
                    self.learner.update(b_s, b_a, b_r, b_s_, b_d, steps)   
                    
            if steps > 10000 and steps % 10000 == 0:
                self._evaluate(steps)
                self._save_model(steps)
                     
            obs = obs_                 
    
    def _evaluate(self, steps):
        success = 0
        for idx in range(self.num_tasks):
            for i in range(self.agent_config.eval_episodes):
                scores = 0
                obs = self.eval_envs.reset(idx)
                success_flag = False
                while True:
                    act = self._action(obs).squeeze(0)
                    obs, reward, done, info = self.eval_envs.step(act, idx)
                    success_flag = success_flag or info['success']
                    scores += reward
                    if done:
                        break
                success += success_flag       
        self.writer.add_scalar("eval-success-steps",success,steps) 
        
    def evaluate(self, load=False, render=False, path=None):
        if load:
            self._load_model(path)
        self.concat_image = np.zeros((150,900,3),np.uint8)    
        success = 0
        expert_rec = []
        attention_rec = []
        query_rec = []
        for idx in range(self.num_tasks):
            for i in range(self.agent_config.eval_episodes):
                scores = 0
                obs = self.eval_envs.reset(idx)
                success_flag = False
                steps = 0
                if self.env_config.name == 'mujoco-mt':    
                    expert_rec_ = []
                    attention_rec_ = []
                    
                while True:
                    steps += 1
                    determinisitc = True
                        
                    act = self._action(obs, determinisitc, False)
                    act = act.squeeze(0)
                    obs, reward, done, info = self.eval_envs.step(act, idx)
                    
                    success_flag = success_flag or info['success']
                    scores += reward
                    
                    if render:
                        self._render(idx,steps)
                    time.sleep(0.01)
                    
                    if done: 
                        expert_rec.append(expert_rec_)
                        attention_rec.append(attention_rec_)
                        
                        success += success_flag
                        print('task:', idx, 'number:', i, 'score:', scores, 'success:', success_flag, 'success_total:', success)
                        break
            if render:
                self.eval_envs.close(idx)
