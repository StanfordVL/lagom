from pathlib import Path
from itertools import count

import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from lagom.utils import pickle_dump
from lagom.utils import set_global_seeds
from lagom.experiment import Config
from lagom.experiment import Grid
from lagom.experiment import Sample
from lagom.experiment import run_experiment

from engine import Engine
from model import VAE
from model import ConvVAE


config = Config(
    {'cuda': True, 
     'log.dir': 'logs/ConvVAE', 
     'log.interval': 100, 
     
     'nn.type': 'ConvVAE',
     'nn.z_dim': 8,
     
     'train.num_epoch': 100,
     'train.batch_size': 128, 
     'eval.batch_size': 128
    })


def make_dataset(config):
    train_dataset = datasets.MNIST('data/', 
                                   train=True, 
                                   download=True, 
                                   transform=transforms.ToTensor())
    test_dataset = datasets.MNIST('data/', 
                                  train=False, 
                                  transform=transforms.ToTensor())
    kwargs = {'num_workers': 1, 'pin_memory': True} if config['cuda'] else {}
    train_loader = DataLoader(train_dataset, 
                              batch_size=config['train.batch_size'], 
                              shuffle=True, 
                              **kwargs)
    test_loader = DataLoader(test_dataset, 
                             batch_size=config['eval.batch_size'], 
                             shuffle=True, 
                             **kwargs)
    return train_loader, test_loader


def run(config, seed, device):
    set_global_seeds(seed)
    logdir = Path(config['log.dir']) / str(config['ID']) / str(seed)

    train_loader, test_loader = make_dataset(config)
    if config['nn.type'] == 'VAE':
        model = VAE(config, device)
    elif config['nn.type'] == 'ConvVAE':
        model = ConvVAE(config, device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    engine = Engine(config, 
                    model=model, 
                    optimizer=optimizer,
                    train_loader=train_loader, 
                    test_loader=test_loader)
    
    train_logs = []
    eval_logs = []
    for epoch in range(config['train.num_epoch']):
        train_logger = engine.train(epoch, logdir=logdir)
        train_logs.append(train_logger.logs)
        eval_logger = engine.eval(epoch, logdir=logdir)
        eval_logs.append(eval_logger.logs)
    pickle_dump(obj=train_logs, f=logdir/'train_logs', ext='.pkl')
    pickle_dump(obj=eval_logs, f=logdir/'eval_logs', ext='.pkl')
    return None


if __name__ == '__main__':
    run_experiment(run=run, 
                   config=config, 
                   seeds=[1770966829], 
                   num_worker=100)
