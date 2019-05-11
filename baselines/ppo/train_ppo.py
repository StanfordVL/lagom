import os
from pathlib import Path
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

from baselines.ppo.agent import Agent
from baselines.ppo.engine import Engine

def runner(config, seed, device, logdir, make_env, args):
    set_global_seeds(seed)

    env = make_env(args)
    #env = env.env
    #print("env: ", env) # TODO - remove debug statement
    #env = TimeLimit(env, env.spec.max_episode_steps)
    #if config['env.clip_action'] and isinstance(env.action_space, Box):
        #env = ClipAction(env)
    
    if config['env.standardize_obs']:
        env = VecStandardizeObservation(env, clip=5.)
    if config['env.standardize_reward']:
        env = VecStandardizeReward(env, clip=10., gamma=config['agent.gamma'])
    env = VecStepInfo(env)
    
    agent = Agent(config, env, device)
    runner = EpisodeRunner(reset_on_call=False)
    engine = Engine(config, agent=agent, env=env, runner=runner, log_dir=logdir)
    train_logs = []
    checkpoint_count = 0
    for i in count():
        if agent.total_timestep >= config['train.timestep']: break
        train_logger = engine.train(i)
        train_logs.append(train_logger.logs)
        if i == 0 or (i+1) % config['log.freq'] == 0:
            train_logger.dump(keys=None, index=0, indent=0, border='-'*50)
        if True: # TODO
        #if agent.total_timestep >= int(config['train.timestep']*(checkpoint_count/(config['checkpoint.num'] - 1))):
            agent.checkpoint(logdir, i + 1)
            checkpoint_count += 1
    pickle_dump(obj=train_logs, f=logdir/'train_logs', ext='.pkl')
    return None
    
def generate_config(args, create_config_obj=True):
    """
    Translate between internal names and lagom-specific names
    """
    config = {'log.freq': 1,
              'checkpoint.num': 1,

              # this is all done internally
              'env.standardize_obs': False,
              'env.standardize_reward': False,
              'env.clip_action': False, 
              
              'nn.sizes': args.num_layers*[args.num_hidden],
              
              'agent.policy_lr': args.learning_rate,
              'agent.use_lr_scheduler': False, 
              'agent.value_lr': args.learning_rate,
              'agent.gamma': args.discount_factor,
              'agent.gae_lambda': args.lam,
              'agent.standardize_adv': True,  # standardize advantage estimates
              'agent.max_grad_norm': args.max_grad_norm, # grad clipping by norm
              'agent.clip_range': args.clip_range,  # ratio clipping
              'agent.std0': 0.6,  # initial std # TODO

              'train.timestep': args.num_timesteps,  # total number of training (environmental) timesteps
              'train.timestep_per_iter': args.nsteps,# number of timesteps per iteration
              'train.batch_size': (args.ncpu * args.nsteps) // args.nminibatches,
              'train.num_epochs': args.n_epochs_per_update
    }
        
    if create_config_obj: 
        return Config(config)
    else:
        return config

def train_ppo(make_env_func, args):
    print("make_env_func: ", make_env_func) # TODO - remove debug statement
    print("args: ", args) # TODO - remove debug statement
    from functools import partial
    config = generate_config(args)
    run_experiment(run=partial(runner, make_env=make_env_func, args=args), 
                   config=config, 
                   seeds=[1770966829],
                   log_dir=os.path.join(args.log_dir, 'lagom'),
                   max_workers=None,# if args.initial_policy is not None else args.ncpu,
                   chunksize=1, 
                   use_gpu=False,  # CPU a bit faster
                   gpu_ids=None)
