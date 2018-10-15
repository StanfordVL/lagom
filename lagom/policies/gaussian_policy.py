from .base_policy import BasePolicy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Independent
from torch.distributions import Normal

from lagom.core.networks import ortho_init


class GaussianPolicy(BasePolicy):
    r"""A parameterized policy defined as independent Gaussian distributions over a continuous action space. 
    
    .. note::
    
        The neural network given to the policy should define all but the final output layer. The final
        output layer for the Gaussian (Normal) distribution will be created with the policy and augmented
        to the network. This decoupled design makes it more flexible to use for different continuous
        action spaces. Note that the network must have an attribute ``.last_feature_dim`` of type
        ``int`` for the policy to create the final output layer (fully-connected), and this is
        recommended to be done in the method :meth:`~BaseNetwork.make_params` of the network class.
        The network outputs the mean :math:`\mu` and log-variance :math:`\log\sigma^2` which allows
        the network to optimize in log-space i.e. negative, zero and positive. 
        
    There are several options for modelling the standard deviation:
    
    * :attr:`min_std` constrains the standard deviation with a lower bound threshould. This helps to avoid
      numerical instability, e.g. producing ``NaN``
        
    * :attr:`std_style` indicates the parameterization of the standard deviation. 

        * For std_style='exp', the standard deviation is obtained by applying exponentiation on log-variance
          i.e. :math:`\exp(0.5\log\sigma^2)`.
        * For std_style='softplus', the standard deviation is obtained by applying softplus operation on
          log-variance, i.e. :math:`f(x) = \log(1 + \exp(x))`.
            
    * :attr:`constant_std` indicates whether to use constant standard deviation or learning it instead.
      If a ``None`` is given, then the standard deviation will be learned. Note that a scalar value
      should be given if using constant value for all dimensions.
        
    * :attr:`std_state_dependent` indicates whether to learn standard deviation with dependency on state.
    
        * For std_state_dependent=True, the log-variance head is created and its forward pass takes
          last feature values as input. 
        * For std_state_dependent=False, the independent trainable nn.Parameter will be created. It
          does not need forward pass, but the backpropagation will calculate its gradients. 
            
    * :attr:`init_std` controls the initial values for independently learnable standard deviation. 
      Note that this is only valid when ``std_state_dependent=False``. 
    
    """
    def __init__(self,
                 config,
                 network, 
                 env_spec, 
                 device,
                 learn_V=False,
                 min_std=1e-6, 
                 std_style='exp', 
                 constant_std=None,
                 std_state_dependent=False,
                 init_std=1.0,
                 **kwargs):
        super().__init__(config=config, network=network, env_spec=env_spec, device=device, **kwargs)
        self.learn_V = learn_V
        
        # Record additional arguments
        self.min_std = min_std
        self.std_style = std_style
        self.constant_std = constant_std
        self.std_state_dependent = std_state_dependent
        self.init_std = init_std
        
        assert self.env_spec.control_type == 'Continuous', 'expected as Continuous control type'
        assert hasattr(self.network, 'last_feature_dim'), 'network expected to have an attribute `.last_feature_dim`'
        if self.constant_std is not None:
            assert not self.std_state_dependent
        
        # Create mean head, orthogonal initialization and put onto device
        mean_head = nn.Linear(in_features=self.network.last_feature_dim, 
                              out_features=self.action_space.flat_dim)
        ortho_init(mean_head, nonlinearity=None, weight_scale=0.01, constant_bias=0.0)  # 0.01->almost zeros initially
        mean_head = mean_head.to(self.device)
        # Augment to network (e.g. tracked by network.parameters() for optimizer to update)
        self.network.add_module('mean_head', mean_head)
        
        # Create logvar head, orthogonal initialization and put onto device
        if self.constant_std is not None:  # using constant std
            if np.isscalar(self.constant_std):  # scalar
                logvar_head = torch.full(size=[self.env_spec.action_space.flat_dim], 
                                         fill_value=torch.log(torch.tensor(self.constant_std)**2),  # log(std**2)
                                         requires_grad=False)  # no grad
            else:  # a numpy array
                logvar_head = torch.log(torch.from_numpy(np.array(self.constant_std)**2).float())
        else:  # no constant std, so learn it
            if self.std_state_dependent:  # state dependent, so a layer
                logvar_head = nn.Linear(in_features=self.network.last_feature_dim, 
                                        out_features=self.env_spec.action_space.flat_dim)
                ortho_init(logvar_head, nonlinearity=None, weight_scale=0.01, constant_bias=0.0)  # 0.01->almost 1.0 std
            else:  # state independent, so a learnable nn.Parameter
                assert self.init_std is not None, f'expected init_std is given as scalar value, got {self.init_std}'
                logvar_head = nn.Parameter(torch.full(size=[self.env_spec.action_space.flat_dim], 
                                                      fill_value=torch.log(torch.tensor(self.init_std)**2), 
                                                      requires_grad=True))  # with grad
        logvar_head = logvar_head.to(self.device)
        # Augment to network as module or as attribute
        if isinstance(logvar_head, nn.Linear):
            self.network.add_module('logvar_head', logvar_head)
        else:
            self.network.logvar_head = logvar_head
            
        # Create value head (if required), orthogonal initialization and put onto device
        if self.learn_V:
            value_head = nn.Linear(in_features=self.network.last_feature_dim, out_features=1)
            ortho_init(value_head, nonlinearity=None, weight_scale=1.0, constant_bias=0.0)
            value_head = value_head.to(self.device)
            self.network.add_module('value_head', value_head)
            
        # Initialize and track the RNN hidden states
        if self.recurrent:
            self.reset_rnn_states()
    
    def __call__(self, x, out_keys=['action'], info={}, **kwargs):
        # Output dictionary
        out_policy = {}
        
        # Forward pass of feature networks to obtain features
        if self.recurrent:
            if 'mask' in info:  # make the mask
                mask = np.logical_not(info['mask']).astype(np.float32)
                mask = torch.from_numpy(mask).unsqueeze(1).to(self.device)
            else:
                mask = None
                
            out_network = self.network(x=x, 
                                       hidden_states=self.rnn_states, 
                                       mask=mask)
            features = out_network['output']
            # Update the tracking of current RNN hidden states
            if 'rnn_state_no_update' not in info:
                self.rnn_states = out_network['hidden_states']
        else:
            features = self.network(x)
        
        # Forward pass through mean head to obtain mean values for Gaussian distribution
        mean = self.network.mean_head(features)
        # Obtain logvar based on the options
        if isinstance(self.network.logvar_head, nn.Linear):  # linear layer, then do forward pass
            logvar = self.network.logvar_head(features)
        else:  # either Tensor or nn.Parameter
            logvar = self.network.logvar_head
            # Expand as same shape as mean
            logvar = logvar.expand_as(mean)
            
        # Forward pass of value head to obtain value function if required
        if 'state_value' in out_keys:
            out_policy['state_value'] = self.network.value_head(features).squeeze(-1)  # squeeze final single dim
        
        # Get std from logvar
        if self.std_style == 'exp':
            std = torch.exp(0.5*logvar)
        elif self.std_style == 'softplus':
            std = F.softplus(logvar)
        
        # Lower bound threshould for std
        min_std = torch.full(std.size(), self.min_std).type_as(std).to(self.device)
        std = torch.max(std, min_std)
        
        # Create independent Gaussian distributions i.e. Diagonal Gaussian
        action_dist = Independent(Normal(loc=mean, scale=std), 1)
        
        # Sample action from the distribution (no gradient)
        # Do not use `rsample()`, it leads to zero gradient of mean head !
        action = action_dist.sample()
        out_policy['action'] = action
        
        # Calculate log-probability of the sampled action
        if 'action_logprob' in out_keys:
            out_policy['action_logprob'] = action_dist.log_prob(action)
        
        # Calculate policy entropy conditioned on state
        if 'entropy' in out_keys:
            out_policy['entropy'] = action_dist.entropy()
        
        # Calculate policy perplexity i.e. exp(entropy)
        if 'perplexity' in out_keys:
            out_policy['perplexity'] = action_dist.perplexity()
        
        # sanity check for NaN
        if torch.any(torch.isnan(action)):
            while True:
                msg = 'NaN ! A workaround is to learn state-independent std or use tanh rather than relu'
                msg2 = f'check: \n\t mean: {mean}, logvar: {logvar}'
                print(msg + msg2)
        
        # Constraint action in valid range
        out_policy['action'] = self.constraint_action(action)
        
        return out_policy
        
    def constraint_action(self, action):
        r"""Clipping the action with valid upper/lower bound defined in action space. 
        
        .. note::
        
            We assume all dimensions in continuous action space share the identical high and low value
            e.g. low = [-2.0, -2.0] and high = [2.0, 2.0]
            
        .. warning::
        
            The constraint action should be placed after computing the log-probability. It happens before
            it, the log-probability will be definitely wrong value. 
        
        Args:
            action (Tensor): action sampled from Normal distribution. 
            
        Returns
        -------
        constrained_action : Tensor
            constrained action.
        """
        # Get valid range
        low = np.unique(self.action_space.low)
        high = np.unique(self.action_space.high)
        assert low.ndim == 1 and high.ndim == 1, 'low and high should be identical for each dimension'
        assert -low.item() == high.item(), 'low and high should be identical with absolute value'
        
        # Clip action to value range i.e. [low, high]
        constrained_action = torch.clamp(action, min=low.item(), max=high.item())
        
        return constrained_action
