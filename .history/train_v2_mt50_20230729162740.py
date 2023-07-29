import torch
import os
import argparse
from agent.mtmhsac_agent import MTMHSAC_Agent
from utils import get_config
from env import MT50_Env
from policy import MixExpertAttentionSACNetwork
from agent import MTMHSAC_Agent

def setup(config_path):
    parser = argparse.ArgumentParser()
    config = get_config(config_path)
    os.environ['DISPLAY'] = ":1"
    torch.random.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    return config

if __name__ == "__main__":
    env_config = setup(config_path="./cfg/mt50.yaml")
    agent_config = setup(config_path="./cfg/mtmesac.yaml")
    # training environment setup    
    envs = MT50_Env(agent_config.task_one_hot, env_config)
    envs.seed(env_config.seed)
    observation_space = envs.observation_space 
    action_space = envs.action_space
    # eval environment setup
    eval_envs = MT50_Env(agent_config.task_one_hot, env_config)
    eval_envs.seed(env_config.seed+100)
    
    policy = MixExpertAttentionSACNetwork(observation_space,
                                        action_space,
                                        env_config.num_tasks,
                                        agent_config)
    agent = MTMHSAC_Agent(env_config,
                        agent_config,
                        envs,
                        eval_envs,
                        policy)
    
    agent.train()