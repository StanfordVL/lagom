from time import perf_counter
from itertools import count, chain

import numpy as np
import torch

from lagom import Logger
from lagom import BaseEngine
from lagom.transform import describe
from lagom.utils import color_str
from lagom.envs.wrappers import get_wrapper

from torch.utils.tensorboard import SummaryWriter

class Engine(BaseEngine):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.writer = SummaryWriter(self.log_dir)
        
    def train(self, n=None, **kwargs):
        
        checkpoint_count = 0
        for i in count():
            if self.agent.total_timestep >= self.config['train.timestep']: break

            # train
            self.agent.train()
            D = self.runner(self.agent, self.env, self.config['train.timestep_per_iter'])
            out_agent = self.agent.learn(D) 

            infos = [info for info in chain.from_iterable([traj.infos for traj in D]) if 'episode' in info]

            def safemean(xs):
                return np.nan if len(xs) == 0 else np.mean(xs)

            self.writer.add_scalar('train_reward_mean', safemean([info['episode']['return'] for info in infos]), i)
            if len(infos) != 0 and 'add_vals' in infos[0]:
                for key in infos[0]['add_vals']:
                    self.writer.add_scalar('train_'+key+'mean', safemean([info[key] for info in infos]), i)

            self.eval(i)
            self.agent.checkpoint(self.log_dir, i + 1)
            checkpoint_count += 1

        return logger
        
    def eval(self, n=None, **kwargs):
        infos = []
        observation = self.eval_env.reset()
        for _ in range(self.eval_env.spec.max_episode_steps):
            with torch.no_grad():
                action = self.agent.choose_action(observation, mode='eval')['raw_action']
            next_observation, reward, done, info = self.eval_env.step(action)
            if done[0]:  # [0] single environment
                infos.append(info[0])
                break
            observation = next_observation
                
        def safemean(xs):
            return np.nan if len(xs) == 0 else np.mean(xs)

        self.writer.add_scalar('eval_reward_mean', np.mean([info['episode']['return'] for info in infos]), n)
        if 'add_vals' in infos[0]:
            for key in infos[0]['add_vals']:
                self.writer.add_scalar('eval_'+key+'mean', safemean([info[key] for info in infos]), n)
        
    def __del__(self):
        self.writer.close()
