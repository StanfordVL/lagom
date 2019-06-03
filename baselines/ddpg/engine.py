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

import logging
logger = logging.getLogger('robosuite.scripts.ddpg.engine')

class Engine(BaseEngine):
    
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.writer = SummaryWriter(self.log_dir)

    def train(self, n=None, **kwargs):
        num_episode = 0
        prev_critic_loss = float('inf')
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

                '''
                # don't update actor if critic loss is too high
                if num_episode < 100 or prev_critic_loss > self.config['agent.critic.burn_in_thresh']:
                #if prev_critic_loss > self.config['agent.critic.burn_in_thresh']:
                    for param_group in self.agent.actor_optimizer.param_groups:
                        param_group['lr'] = 0
                # but put back the learning rate if the loss is sufficiently low
                elif self.agent.actor_optimizer.param_groups[0]['lr'] == 0:
                    for param_group in self.agent.actor_optimizer.param_groups:
                        param_group['lr'] = self.config['agent.critic.lr']
                        logger.info("Just updated learning rate")
                '''
                
                # updates in the end of episode, for each time step
                out_agent = self.agent.learn(D=None, replay=self.replay, episode_length=step_info[0]['episode']['horizon'])

                prev_critic_loss = out_agent['critic_loss']
                num_episode += 1
                
                self.writer.add_scalar('actor_lr', self.agent.actor_optimizer.param_groups[0]['lr'], num_episode)
                self.writer.add_scalar('critic_loss', out_agent['critic_loss'], num_episode)
                self.writer.add_scalar('actor_loss', out_agent['actor_loss'], num_episode)

                self.agent.checkpoint(self.log_dir, num_episode)
                self.eval(num_episode)
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
        self.writer.add_scalar('eval_reward_mean', safemean([info['episode']['return'] for info in infos]), n)
        if 'add_vals' in infos[0]:
            for key in infos[0]['add_vals']:
                self.writer.add_scalar('eval_'+key+'mean', safemean([info[key] for info in infos]), n)
        
        return None
        
    def __del__(self):
        self.writer.close()

