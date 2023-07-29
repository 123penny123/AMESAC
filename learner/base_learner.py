import torch
import time
from abc import ABC,abstractmethod
from typing import Optional,Union
from torch.utils.tensorboard import SummaryWriter
class Learner(ABC):
    def __init__(self,
                 policy:torch.nn.Module,
                 optimizer:torch.optim.Optimizer,
                 scheduler:Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 summary_writer:Optional[SummaryWriter] = None,
                 device:Optional[Union[int,str,torch.device]]=None,
                 modeldir:str = "./"):
        self.policy = policy
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = summary_writer
        self.device = device
        self.modeldir = modeldir
        self.iterations = 0
    def save_model(self):
        time_string = time.asctime()
        time_string = time_string.replace(" ","")
        model_path = self.modeldir + "model-%s-%s.pth"%(time.asctime(),str(self.iterations))
        torch.save(self.policy,model_path)
    def load_model(self,path):
        model_path = self.modeldir + path
        model_dict = torch.load(model_path)
        self.policy.load_state_dict(model_dict)
    @abstractmethod
    def update(self,*args):
        raise NotImplementedError