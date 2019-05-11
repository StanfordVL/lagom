from time import perf_counter
from itertools import chain

import numpy as np

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
        self.agent.train()
        start_time = perf_counter()
        
        D = self.runner(self.agent, self.env, self.config['train.timestep_per_iter'])
        out_agent = self.agent.learn(D) 
        
        logger = Logger()
        logger('train_iteration', n+1)
        logger('num_seconds', round(perf_counter() - start_time, 1))
        [logger(key, value) for key, value in out_agent.items()]
        logger('num_trajectories', len(D))
        logger('num_timesteps', sum([len(traj) for traj in D]))
        logger('accumulated_trained_timesteps', self.agent.total_timestep)
        G = [traj.numpy_rewards.sum() for traj in D]
        logger('return', describe(G, axis=-1, repr_indent=1, repr_prefix='\n'))
        
        infos = [info for info in chain.from_iterable([traj.infos for traj in D]) if 'episode' in info]
        online_returns = [info['episode']['return'] for info in infos]
        online_horizons = [info['episode']['horizon'] for info in infos]
        logger('online_return', describe(online_returns, axis=-1, repr_indent=1, repr_prefix='\n'))
        logger('online_horizon', describe(online_horizons, axis=-1, repr_indent=1, repr_prefix='\n'))
            
        monitor_env = get_wrapper(self.env, 'VecMonitor')
        logger('running_return', describe(monitor_env.return_queue, axis=-1, repr_indent=1, repr_prefix='\n'))
        logger('running_horizon', describe(monitor_env.horizon_queue, axis=-1, repr_indent=1, repr_prefix='\n'))

        def safemean(xs):
            return np.nan if len(xs) == 0 else np.mean(xs)

        for key in infos[0]['add_vals']:
            self.writer.add_scalar(key+'_mean', safemean([info[key] for info in infos]), n)
        
        return logger
        
    def eval(self, n=None, **kwargs):
        pass

    def __del__(self):
        self.writer.close()
