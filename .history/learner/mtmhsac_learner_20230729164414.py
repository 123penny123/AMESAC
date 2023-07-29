from copy import deepcopy
import itertools
import torch 
import time
import torch.nn as nn
import torch.nn.functional as F
from .base_learner import Learner
from typing import Optional,Union
from gym.spaces import Space
from torch.utils.tensorboard import SummaryWriter

q_max_norm = 20 
pi_max_norm = 20 
alpha_max_norm = 10

class MTMHSAC_Learner(Learner):
    def __init__(self,
                 action_space: Space, 
                 policy:nn.Module,
                 summary_writer:Optional[SummaryWriter] = None,
                 device: Optional[Union[int,str,torch.device]] = None,
                 num_tasks: int=0,
                 modeldir: str = "./",
                 policy_type="Gaussian",
                 automatic_entropy_tuning=True, 
                 loss_reweight = True,
                 tau: float=0.005,
                 alpha: float=0.2,
                 gamma: float=0.99,
                 lr: float=0.0003,
                 batch_size=1024,
                 policy_delay=2,
                 betas=[0.9, 0.999]):
        self.policy = policy
        self.policy_tar = deepcopy(policy)
        
        for p in self.policy_tar.parameters():
            p.requires_grad = False
        
        self.tau = tau
        self.alpha = torch.as_tensor([alpha] * num_tasks, device=device)
        self.gamma = gamma
        self.writer = summary_writer
        self.device = device
        self.num_tasks = num_tasks
        self.modeldir = modeldir
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.update_cnt = 0
        self.batch_size = batch_size
        self.policy_delay = policy_delay
        self.loss_reweight = loss_reweight
        
        self.q_params = itertools.chain(self.policy.q1.parameters(), self.policy.q2.parameters())
        self.q_optimizer = torch.optim.Adam(self.q_params, lr=lr)
        self.pi_optimizer = torch.optim.Adam(self.policy.pi.parameters(), lr=lr)
        
        self.reweight = torch.ones(self.num_tasks*self.batch_size, requires_grad=True, device=self.device)

        if policy_type == "Gaussian":
            self.log_alpha = []
            self.alpha_optimizer = []
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(self.num_tasks, requires_grad=True, device=self.device)
            self.alpha_optimizer = (torch.optim.Adam([self.log_alpha], lr=lr))
            self.alpha_params = itertools.chain([self.log_alpha])

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
        
    def compute_loss_q(self, batch_s, batch_a, batch_r, batch_s_, batch_d, update_cnt):
        
        batch_a = torch.as_tensor(batch_a,device=self.device)
        batch_r = torch.as_tensor(batch_r,device=self.device)
        batch_d = torch.as_tensor(batch_d,device=self.device)
        batch_size = int(batch_s.shape[0]/self.num_tasks)
        
        q1, expert_loss1 = self.policy.q1(batch_s,batch_a)
        q2, expert_loss2 = self.policy.q2(batch_s,batch_a)
        expert_loss = 0.5*(expert_loss1+expert_loss2)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            next_action, logp_a2,_ = self.policy.pi(batch_s_)

            # Target Q-values
            q1_pi_targ,_ = self.policy_tar.q1(batch_s_, next_action)
            q2_pi_targ,_ = self.policy_tar.q2(batch_s_, next_action)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            alpha = self.alpha.unsqueeze(-1).expand(self.alpha.shape[-1], batch_size).reshape(-1).detach()  
            backup = batch_r + self.gamma * (1 - batch_d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = (q1 - backup)**2*self.reweight
        loss_q2 = (q2 - backup)**2*self.reweight
        
        if update_cnt % 20 == 0:
            for idx in range (self.num_tasks):
                self.writer.add_scalars("train-lossq-steps",{"env-%d"%idx:loss_q1[idx*batch_size:(idx+1)*batch_size].mean().item()},update_cnt)
    
        loss_q1 = loss_q1.mean()
        loss_q2 = loss_q2.mean()
        loss_q = loss_q1 + loss_q2 #+ expert_loss
        
        return loss_q 

    # Set up function for computing SAC pi loss
    def compute_loss_pi_and_alpha(self, batch_s, update_cnt):
        batch_size = int(batch_s.shape[0]/self.num_tasks)

        pi, logp_pi, expert_loss = self.policy.pi(batch_s)
        q1_pi,_ = self.policy.q1(batch_s, pi)
        q2_pi,_ = self.policy.q2(batch_s, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        
        alpha = self.alpha.unsqueeze(-1).expand(self.alpha.shape[-1], batch_size).reshape(-1).detach()
        loss_pi = (alpha * logp_pi - q_pi)*self.reweight 
        
        if self.automatic_entropy_tuning:   # adjust temperature
            log_alpha = self.log_alpha.unsqueeze(-1).expand(self.log_alpha.shape[-1], batch_size).reshape(-1)
            # for mujoco-mt
            # target_entropy = torch.tensor([-8,-6,-3,-6],device=self.device).unsqueeze(-1).tile(1,self.batch_size).flatten()
            alpha_loss = -(log_alpha * (logp_pi + self.target_entropy).detach())*self.reweight
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            
        if update_cnt % 20 == 0:
            for idx in range(self.num_tasks):
                self.writer.add_scalars("train-losspi-steps",{"env-%d"%idx:loss_pi[idx*batch_size:(idx+1)*batch_size].mean().item()},update_cnt)
                self.writer.add_scalars("train-lossalpha-steps",{"env-%d"%idx:alpha_loss[idx*batch_size:(idx+1)*batch_size].mean().item()},update_cnt)
        
        loss_pi = loss_pi.mean() #+ expert_loss
        alpha_loss = alpha_loss.mean()
        
        return loss_pi, alpha_loss

    def update(self, b_s, b_a, b_r, b_s_, b_d, update_cnt):
        if self.loss_reweight:
            self.reweight = (F.softmax(-self.alpha).unsqueeze(-1).tile(1,self.batch_size).flatten())*4
        
        loss_q = self.compute_loss_q(b_s, b_a, b_r, b_s_, b_d, update_cnt)
                
        self.q_optimizer.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(self.q_params, q_max_norm)
        self.q_optimizer.step()
        
        if update_cnt % self.policy_delay == 0:
            for p in self.q_params:
                p.requires_grad = False
                
            loss_pi, alpha_loss = self.compute_loss_pi_and_alpha(b_s, update_cnt)
                
            self.pi_optimizer.zero_grad()
            loss_pi.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.pi.parameters(), pi_max_norm)
            self.pi_optimizer.step()
                
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.alpha_params, alpha_max_norm)
            self.alpha_optimizer.step()
            
            for p in self.q_params:
                p.requires_grad = True
                
            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                self.soft_update(self.policy_tar, self.policy, self.tau)
        
            for idx in range(self.num_tasks):
                self.alpha[idx] = self.log_alpha[idx].exp().detach()
                reweight = F.softmax(-self.alpha)*4
                if update_cnt % 20 == 0:  
                    self.writer.add_scalars("lr-weight",{"env-%d"%idx:reweight[idx].item()},update_cnt) 
                    self.writer.add_scalars("alpha",{"env-%d"%idx:self.alpha[idx].item()},update_cnt)
                    
            if update_cnt % 20 == 0:        
                self.writer.add_scalar("total_q_loss",loss_q.item(),update_cnt)
                self.writer.add_scalar("total_pi_loss",loss_pi.item(),update_cnt)
                self.writer.add_scalar("total_alpha_loss",alpha_loss.item(),update_cnt)