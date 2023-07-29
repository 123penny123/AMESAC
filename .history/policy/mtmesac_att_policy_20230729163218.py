import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

torch.manual_seed(1996)

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
def mlp(sizes, activation, device, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1], device=device), act()]
    return nn.Sequential(*layers)

class representation(nn.Module):
    def __init__(self,
                 obs_dim,
                 act_dim,
                 config,
                 activation,
                 device,
                 num_tasks,
                 act=False):
        super().__init__()
        self.device = device
        self.config = config
        self.num_tasks = num_tasks
        self.act_dim = act_dim
        self.act = act       
        
        self.state_rep = mlp([obs_dim] + list(config.rep_hidden_sizes), activation, device)
        if act:
            self.state_rep = mlp([obs_dim+act_dim] + list(config.rep_hidden_sizes), activation, device)
        self.embedding = nn.Parameter(0.5*torch.ones((self.num_tasks,config.exp_hidden_sizes[-1]),device=device),requires_grad=True)    ################
        
    def forward(self, state,act=None):
        task_embedding = state[:, -self.num_tasks:]
        state = state[:, :-self.num_tasks]
        # de-one-hot
        task_embedding = torch.nonzero(task_embedding)[:,-1].unsqueeze(-1) 
        task_embedding = F.tanh(self.embedding[task_embedding])
        if self.act:
            rep = self.state_rep(torch.cat([state,act],dim=-1))
        else:
            rep = self.state_rep(state)
        embedding_query = task_embedding
        return rep, embedding_query

class MixExpertAttentionQNetwork(nn.Module):
    def __init__(self, 
                 obs_dim,
                 act_dim, 
                 config, 
                 activation,
                 representation,
                 device,
                 num_tasks):
        super().__init__()        
        self.device = device
        self.config = config
        self.num_tasks = num_tasks
        self.representation = representation
        self.expert_key = nn.ModuleList()
        self.expert_value = nn.ModuleList()
        for _ in range(config.expert_num):
            self.expert_key.append(mlp(config.exp_hidden_sizes,activation,device)) 
            self.expert_value.append(mlp(config.exp_value_hidden_sizes,activation,device))  
        self.tower = mlp(config.tow_hidden_sizes + [256] + [1],activation,device) 

    def forward(self, state, act):               
        state = torch.as_tensor(state,device=self.device,dtype=torch.float32)
        if len(state.shape)==1:
            state = state.unsqueeze(0) 
        act = torch.as_tensor(act,device=self.device,dtype=torch.float32)
        task_embedding = state[:, -self.num_tasks:]
        
        rep,embedding_query = self.representation(state,act)
        expert_key = []
        expert_value = []
        attention = []
        for i in range(self.config.expert_num):
            query = embedding_query[:,0,:]
            expert_key.append(self.expert_key[i](rep))
            expert_value.append(self.expert_value[i](rep).view(-1,1,self.config.exp_hidden_sizes[-1]))  
            attention.append((query * expert_key[i]).sum(-1).unsqueeze(-1))
        # attention calculation
        attention = torch.cat(attention, dim=-1)
        attention = nn.Softmax(dim=1)(attention)
        weight = attention.unsqueeze(1)
        expert_value = torch.cat(expert_value,dim=1)
        tower_input = torch.bmm(weight,expert_value) 
        tower_input = tower_input.squeeze(1)
        # de-one-hot
        task_embedding = torch.nonzero(task_embedding)[:,-1].unsqueeze(-1)
        q = self.tower(tower_input)

        expert_loss = -torch.mean(torch.squeeze((torch.clip(torch.log(weight+1e-10),-6,0)).sum(-1)*0.3, -1))
        return torch.squeeze(q, -1), expert_loss
    
class MixExpertAttentionGaussianPolicy(nn.Module):
    def __init__(self, 
                obs_dim, 
                act_dim, 
                activation, 
                representation,
                act_limit,
                config,
                device,
                num_tasks):
        super().__init__()        
        self.device = device
        self.config = config
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.LOG_SIG_MIN = config.LOG_SIG_MIN
        self.LOG_SIG_MAX = config.LOG_SIG_MAX
        self.epsi = config.epsi
        self.num_tasks = num_tasks
        self.expert_key = nn.ModuleList()
        self.expert_value = nn.ModuleList()  
        self.representation = representation
        for _ in range(config.expert_num):
            self.expert_key.append(mlp(config.exp_hidden_sizes,activation,device))
            self.expert_value.append(mlp(config.exp_value_hidden_sizes,activation,device))    
        self.mu_layer = mlp(config.tow_hidden_sizes + [act_dim],activation,device) 
        self.log_std_layer = mlp(config.tow_hidden_sizes + [act_dim],activation,device)
        
    def entropy(self):
        return self.pi_distribution.entropy().sum(-1)
    
    def forward(self, state, deterministic=False, with_logprob=True): 
        state = torch.as_tensor(state,device=self.device,dtype=torch.float32)
        if len(state.shape)==1:
            state = state.unsqueeze(0)   
        task_embedding = state[:, -self.num_tasks:]
        
        rep,embedding_query = self.representation(state)
        expert_key = []
        expert_value = []
        attention = []
        # data representation and augmentation forward
        for i in range(self.config.expert_num):
            query = embedding_query[:,0,:]
            expert_key.append(self.expert_key[i](rep))
            expert_value.append(self.expert_value[i](rep).view(-1,1,self.config.exp_hidden_sizes[-1]))
            attention.append((query * expert_key[i]).sum(-1).unsqueeze(-1))
        # attention calculation
        attention = torch.cat(attention, dim=-1)
        attention = nn.Softmax(dim=1)(attention)
        weight = attention.unsqueeze(1)
        expert_value = torch.cat(expert_value,dim=1)
        tower_input = torch.bmm(weight,expert_value) 
        tower_input = tower_input.squeeze(1)
        
        # de-one-hot
        task_embedding = torch.nonzero(task_embedding)[:,-1].unsqueeze(-1) * self.act_dim
        index = torch.as_tensor(list(range(0, self.act_dim)), device=self.device)
        task_embedding = task_embedding.expand(state.shape[0], self.act_dim) + index
       
        mu = self.mu_layer(tower_input)
        log_std = self.log_std_layer(tower_input)
        log_std = torch.clamp(log_std, self.LOG_SIG_MIN, self.LOG_SIG_MAX)
        std = torch.exp(log_std) 

        # Pre-squash distribution and sample
        self.pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            # re-parameterization  pi-action = mean + N(0,1) * std
            pi_action = self.pi_distribution.rsample()  

        if with_logprob:
            logp_pi = self.pi_distribution.log_prob(pi_action).sum(axis=-1)
            squashed_action = torch.tanh(pi_action)
            logp_pi -= torch.log((1 - squashed_action.pow(2)) + self.epsi).sum(axis=-1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        expert_loss = -torch.mean(torch.squeeze((torch.clip(torch.log(weight+1e-10),-6,0)).sum(-1)*0.3, -1))

        return pi_action, logp_pi, expert_loss

class MixExpertAttentionSACNetwork(nn.Module):

    def __init__(self, 
                 obs_space,
                 action_space,
                 num_tasks,
                 agent_config,
                 activation=nn.ReLU):
        super().__init__()
        obs_dim = obs_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        device = agent_config.device

        q1_rep = representation(obs_dim,act_dim,agent_config,activation,device,num_tasks,act=True)
        q2_rep = representation(obs_dim,act_dim,agent_config,activation,device,num_tasks,act=True)
        pi_rep = representation(obs_dim,act_dim,agent_config,activation,device,num_tasks)
        # build policy and value functions
        self.pi = MixExpertAttentionGaussianPolicy(obs_dim, 
                                        act_dim, 
                                        activation, 
                                        pi_rep,
                                        act_limit,
                                        agent_config,
                                        device,
                                        num_tasks)
        self.q1 = MixExpertAttentionQNetwork(obs_dim,
                                            act_dim, 
                                            agent_config, 
                                            activation,
                                            q1_rep,
                                            device,
                                            num_tasks)
        self.q2 = MixExpertAttentionQNetwork(obs_dim,
                                            act_dim, 
                                            agent_config, 
                                            activation,
                                            q2_rep,
                                            device,
                                            num_tasks)
        self.apply(weights_init_)

    def act(self, state, deterministic=False, plot=False):        
        with torch.no_grad():
            a, _, _ = self.pi(state, deterministic, False)
            return a.detach().cpu().numpy()
