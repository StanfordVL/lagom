import os
from pathlib import Path
from functools import partial
from itertools import count
import gym

from lagom.utils import pickle_dump
from lagom.utils import set_global_seeds
from lagom.experiment import Config
from lagom.experiment import Grid
from lagom.experiment import Sample
from lagom.experiment import Condition
from lagom.experiment import run_experiment
from lagom.envs import make_vec_env
from lagom.envs.wrappers import TimeLimit
from lagom.envs.wrappers import ClipAction
from lagom.envs.wrappers import VecMonitor

from .agent import Agent
from .engine import Engine
from .replay_buffer import ReplayBuffer

def runner(config, seed, device, logdir, make_env, args):
    set_global_seeds(seed)

    print("Now running....")
    env = make_env(args)
    eval_env = make_env(args)
    agent = Agent(config, env, device)
    replay = ReplayBuffer(config['replay.capacity'], device)
    engine = Engine(config, agent=agent, env=env, eval_env=eval_env, replay=replay)

    train_logs = []
    checkpoint_count = 0
    for i in count():
        if agent.total_timestep >= config['train.timestep']: break
        train_logger = engine.train(i)
        train_logs.append(train_logger.logs)
        if i == 0 or (i+1) % config['log.freq'] == 0:
            train_logger.dump(keys=None, index=0, indent=0, border='-'*50)
        if True: 
            agent.checkpoint(logdir, i + 1)
            checkpoint_count += 1
    
    print("train_logs: ", train_logs) # TODO - remove debug statement
    #pickle_dump(obj=train_logs, f=logdir/'train_logs', ext='.pkl')
    #pickle_dump(obj=eval_logs, f=logdir/'eval_logs', ext='.pkl')
    return None  

def train_td3(make_env_func, args):

    config = Config(
        {'cuda': True, 
         'log.freq': 1,#5,  # every n episodes
         'checkpoint.freq': 10,#int(1e5),  # every n timesteps

         #'env.id': Grid(['HalfCheetah-v3', 'Hopper-v3', 'Walker2d-v3', 'Swimmer-v3']),

         'agent.gamma': 0.99,
         'agent.polyak': 0.995,  # polyak averaging coefficient for targets update
         'agent.actor.lr': 1e-3, 
         'agent.actor.use_lr_scheduler': False,
         'agent.critic.lr': 1e-3,
         'agent.critic.use_lr_scheduler': False,
         'agent.action_noise': 0.1,
         'agent.target_noise': 0.2,
         'agent.target_noise_clip': 0.5,
         'agent.policy_delay': 2,
         'agent.max_grad_norm': 999999,  # grad clipping by norm

         'replay.capacity': 1000000, 
         # number of time steps to take uniform actions initially
         'replay.init_size': 10000,#Condition(lambda x: 1000 if x['env.id'] in ['Hopper-v3', 'Walker2d-v3'] else 10000),  
         'replay.batch_size': 100,

         'train.timestep': int(1e6),  # total number of training (environmental) timesteps
         'eval.freq': 128,#5000,
         'eval.num_episode': 1#10

        })


    # Note: this must be a partial to allow passing in a function to runner
    # runner cannot be nested here because then the multiprocessing code would not be able to pickle it
    run_experiment(run=partial(runner, make_env=make_env_func, args=args), 
                   config=config, 
                   seeds=[4153361530],
                   log_dir=os.path.join(args.log_dir, 'lagom'),
                   max_workers=args.ncpu)
    
    return agent
