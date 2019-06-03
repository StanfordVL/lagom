import os
from pathlib import Path
from functools import partial
from itertools import count

import gym
from gym.spaces import Box

from lagom.utils import pickle_dump
from lagom.utils import set_global_seeds
from lagom.experiment import Config
from lagom.experiment import Grid
from lagom.experiment import Sample
from lagom.experiment import run_experiment
from lagom.envs import make_vec_env
from lagom.envs.wrappers import TimeLimit
from lagom.envs.wrappers import ClipAction
from lagom.envs.wrappers import VecMonitor
from lagom.envs.wrappers import VecStandardizeObservation
from lagom.envs.wrappers import VecStandardizeReward
from lagom.envs.wrappers import VecStepInfo
from lagom.runner import EpisodeRunner

from .agent import Agent
from .engine import Engine
from .replay_buffer import ReplayBuffer

def runner(config, seed, device, logdir, make_env, args):
    set_global_seeds(seed)

    env = make_env(args)
    args.replay = True
    eval_env = make_env(args)
    
    agent = Agent(config, env, device)
    replay = ReplayBuffer(env, config['replay.capacity'], device)
    engine = Engine(config, agent=agent, env=env, eval_env=eval_env, replay=replay, log_dir=logdir)
    engine.train()
    
    return None

def generate_config(args, create_config_obj=True):
    """
    Translate between internal names and lagom-specific names
    """
    config = {'log.freq': 1,  # every n timesteps
              'checkpoint.num': 1,
     
              'agent.gamma': args.gamma,
              'agent.polyak': args.polyak,  # polyak averaging coefficient for targets update
              'agent.actor.lr': args.actor_lr, 
              'agent.actor.use_lr_scheduler': args.actor_use_lr_scheduler,
              'agent.critic.lr': args.critic_lr,
              'agent.critic.use_lr_scheduler': args.critic_use_lr_scheduler,
              'agent.initial_temperature': args.initial_temperature,
              'agent.max_grad_norm': args.max_grad_norm,  # grad clipping by norm
              
              'replay.capacity': args.replay_capacity, 
              # number of time steps to take uniform actions initially
              'replay.init_size': args.replay_init_size,
              'replay.batch_size': args.replay_batch_size,
              
              'train.timestep': args.num_timesteps,  # total number of training (environmental) timesteps
              'eval.freq': 1,
              'eval.num_episode': 1
    }

    if create_config_obj: 
        return Config(config)
    else:
        return config
    
def train_sac(make_env_func, args):
    # Note: this must be a partial to allow passing in a function to runner
    # runner cannot be nested here because then the multiprocessing code would not be able to pickle it
    config = generate_config(args)
    run_experiment(run=partial(runner, make_env=make_env_func, args=args), 
                   config=config, 
                   seeds=[args.seed],
                   log_dir=os.path.join(args.log_dir, 'lagom'),
                   max_workers=None, #args.ncpu,
                   chunksize=1,
                   use_gpu=False # TODO - try GPU
    )
    
    return agent
