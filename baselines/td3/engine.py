from time import perf_counter
from itertools import count

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
        eval_togo = 0
        num_episode = 0
        observation, _ = self.env.reset()
        for i in count():
            if i >= self.config['train.timestep']:
                break
            if i < self.config['replay.init_size']:
                action = [self.env.action_space.sample()]
            else:
                action = self.agent.choose_action(observation, mode='train')['action']
            next_observation, reward, step_info = self.env.step(action)
            if step_info[0].last:  # [0] due to single environment
                self.replay.add(observation[0], action[0], reward[0], step_info[0]['last_observation'], step_info[0].terminal)
                
                # updates in the end of episode, for each time step
                out_agent = self.agent.learn(D=None, replay=self.replay, episode_length=step_info[0]['episode']['horizon'])
                num_episode += 1
                if True:
                    self.agent.checkpoint(self.log_dir, num_episode)
                if eval_togo >= self.config['eval.freq']:
                    eval_togo %= self.config['eval.freq']
                    eval_logs.append(self.eval(accumulated_trained_timesteps=(i+1), 
                                               accumulated_trained_episodes=num_episode))
            else:
                self.replay.add(observation[0], action[0], reward[0], next_observation[0], step_info[0].terminal)
            observation = next_observation

        return train_logs, eval_logs

    def eval(self, n=None, **kwargs):
        infos = []
        for _ in range(self.config['eval.num_episode']):
            observation = self.eval_env.reset()
            for _ in range(self.eval_env.spec.max_episode_steps):
                with torch.no_grad():
                    action = self.agent.choose_action(observation, mode='eval')['action']
                next_observation, reward, done, info = self.eval_env.step(action)
                if done[0]:  # [0] single environment
                    infos.append(info[0])
                    break
                observation = next_observation
                
        def safemean(xs):
            return np.nan if len(xs) == 0 else np.mean(xs)
    
        self.writer.add_scalar('eval_reward_mean', np.mean([info['episode']['return'] for info in infos]), n)
        for key in infos[0]['add_vals']:
            self.writer.add_scalar(key+'_mean', safemean([info[key] for info in infos]), n)
        
        return None
        
    def __del__(self):
        self.writer.close()
