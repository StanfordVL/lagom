import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from .base_agent import BaseAgent

from lagom.core.transform import Standardize


class A2CAgent(BaseAgent):
    r"""`Advantage Actor-Critic`_ (A2C) with option to use Generalized Advantage Estimate (GAE)
    
    The main difference of A2C is to use bootstrapping for estimating the advantage function and training value function. 
    
    .. _Advantage Actor-Critic:
        https://arxiv.org/abs/1602.01783
    
    Like `OpenAI baselines` we use fixed-length segments of experiment to compute returns and advantages. 
    
    .. _OpenAI baselines:
        https://blog.openai.com/baselines-acktr-a2c/
    
    .. note::
    
        Use :class:`SegmentRunner` to collect data, not :class:`TrajectoryRunner`
    
    """
    def __init__(self, config, device, policy, optimizer, **kwargs):
        self.policy = policy
        self.optimizer = optimizer
        
        super().__init__(config, device, **kwargs)
        
        # accumulated trained timesteps
        self.total_T = 0
        
    def choose_action(self, obs, info={}):
        if not torch.is_tensor(obs):
            obs = np.asarray(obs)
            assert obs.ndim >= 2, f'expected at least 2-dim for batched data, got {obs.ndim}'
            obs = torch.from_numpy(obs).float().to(self.device)
            
        if self.policy.recurrent and self.info['reset_rnn_states']:
            self.policy.reset_rnn_states(batch_size=obs.size(0))
            self.info['reset_rnn_states'] = False  # Done, turn off
            
        out_policy = self.policy(obs, 
                                 out_keys=['action', 'action_logprob', 'state_value', 
                                           'entropy', 'perplexity'], 
                                 info=info)
        
        return out_policy
        
    def learn(self, D, info={}):
        batch_policy_loss = []
        batch_entropy_loss = []
        batch_value_loss = []
        batch_total_loss = []
        
        for segment in D:
            logprobs = segment.all_info('action_logprob')
            entropies = segment.all_info('entropy')
            Qs = segment.all_bootstrapped_discounted_returns
            
            # Standardize: encourage/discourage half of performed actions
            if self.config['agent.standardize_Q']:
                Qs = Standardize()(Qs).tolist()
                
            # State values
            Vs, finals = segment.all_V
            final_Vs, final_dones = zip(*finals)
            assert len(Vs) == len(segment.transitions)
                
            # Advantage estimates
            As = [Q - V.item() for Q, V in zip(Qs, Vs)]    
            if self.config['agent.standardize_adv']:
                As = Standardize()(As).tolist()
            
            # Estimate policy gradient for all time steps and record all losses
            policy_loss = []
            entropy_loss = []
            value_loss = []
            for logprob, entropy, A, Q, V in zip(logprobs, entropies, As, Qs, Vs):
                policy_loss.append(-logprob*A)
                entropy_loss.append(-entropy)
                value_loss.append(F.mse_loss(V, torch.tensor(Q).view_as(V).to(V.device)))
            for final_V, final_done in zip(final_Vs, final_dones):  # learn terminal state value as zero
                if final_done:
                    value_loss.append(F.mse_loss(final_V, torch.tensor(0.0).view_as(V).to(V.device)))
        
            # Average losses over all time steps
            policy_loss = torch.stack(policy_loss).mean()
            entropy_loss = torch.stack(entropy_loss).mean()
            value_loss = torch.stack(value_loss).mean()
        
            # Calculate total loss
            entropy_coef = self.config['agent.entropy_coef']
            value_coef = self.config['agent.value_coef']
            total_loss = policy_loss + value_coef*value_loss + entropy_coef*entropy_loss
        
            # Record all losses
            batch_policy_loss.append(policy_loss)
            batch_entropy_loss.append(entropy_loss)
            batch_value_loss.append(value_loss)
            batch_total_loss.append(total_loss)
        
        # Average loss over list of Segment
        policy_loss = torch.stack(batch_policy_loss).mean()
        entropy_loss = torch.stack(batch_entropy_loss).mean()
        value_loss = torch.stack(batch_value_loss).mean()
        loss = torch.stack(batch_total_loss).mean()
        
        # Train with estimated policy gradient
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.config['agent.max_grad_norm'] is not None:
            clip_grad_norm_(parameters=self.policy.network.parameters(), 
                            max_norm=self.config['agent.max_grad_norm'], 
                            norm_type=2)
        
        if hasattr(self, 'lr_scheduler'):
            if 'train.iter' in self.config:  # iteration-based
                self.lr_scheduler.step()
            elif 'train.timestep' in self.config:  # timestep-based
                self.lr_scheduler.step(self.total_T)
            else:
                raise KeyError('expected `train.iter` or `train.timestep` in config, but got none of them')
        
        self.optimizer.step()
        
        # Accumulate trained timesteps
        self.total_T += sum([segment.T for segment in D])
        
        out = {}
        out['loss'] = loss.item()
        out['policy_loss'] = policy_loss.item()
        out['entropy_loss'] = entropy_loss.item()
        out['value_loss'] = value_loss.item()
        if hasattr(self, 'lr_scheduler'):
            out['current_lr'] = self.lr_scheduler.get_lr()

        return out
    
    def save(self, f):
        self.policy.network.save(f)
    
    def load(self, f):
        self.policy.network.load(f)
