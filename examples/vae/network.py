import torch
import torch.nn as nn
import torch.nn.functional as F

from lagom.networks import BaseNetwork
from lagom.networks import make_fc
from lagom.networks import make_cnn
from lagom.networks import make_transposed_cnn
from lagom.networks import ortho_init


class Encoder(BaseNetwork):
    def make_params(self, config):
        self.feature_layers = make_fc(784, [400])
        
        self.mean_head = nn.Linear(400, config['network.z_dim'])
        self.logvar_head = nn.Linear(400, config['network.z_dim'])
        
    def init_params(self, config):
        for layer in self.feature_layers:
            ortho_init(layer, nonlinearity='relu', constant_bias=0.0)
            
        ortho_init(self.mean_head, weight_scale=0.01, constant_bias=0.0)
        ortho_init(self.logvar_head, weight_scale=0.01, constant_bias=0.0)
        
    def reset(self, config, **kwargs):
        pass
        
    def forward(self, x):
        x = x.flatten(start_dim=1)
        
        for layer in self.feature_layers:
            x = F.relu(layer(x))
            
        mu = self.mean_head(x)
        logvar = self.logvar_head(x)
        
        return mu, logvar
    

class Decoder(BaseNetwork):
    def make_params(self, config):
        self.feature_layers = make_fc(config['network.z_dim'], [400])
        
        self.x_head = nn.Linear(400, 784)
        
    def init_params(self, config):
        for layer in self.feature_layers:
            ortho_init(layer, nonlinearity='relu', constant_bias=0.0)
            
        ortho_init(self.x_head, nonlinearity='sigmoid', constant_bias=0.0)
        
    def reset(self, config, **kwargs):
        pass
    
    def forward(self, x):
        for layer in self.feature_layers:
            x = F.relu(layer(x))
            
        x = torch.sigmoid(self.x_head(x))
        
        return x
    
    
class ConvEncoder(BaseNetwork):
    def make_params(self, config):
        self.feature_layers = make_cnn(input_channel=1, 
                                       channels=[64, 64, 64], 
                                       kernels=[4, 4, 4], 
                                       strides=[2, 2, 1], 
                                       paddings=[0, 0, 0])
        
        self.mean_head = nn.Linear(256, config['network.z_dim'])
        self.logvar_head = nn.Linear(256, config['network.z_dim'])
        
    def init_params(self, config):
        for layer in self.feature_layers:
            ortho_init(layer, nonlinearity='relu', constant_bias=0.0)
        
        ortho_init(self.mean_head, weight_scale=0.01, constant_bias=0.0)
        ortho_init(self.logvar_head, weight_scale=0.01, constant_bias=0.0)
        
    def reset(self, config, **kwargs):
        pass
    
    def forward(self, x):
        for layer in self.feature_layers:
            x = F.relu(layer(x))
        
        # To shape [N, D]
        x = x.flatten(start_dim=1)
        
        mu = self.mean_head(x)
        logvar = self.logvar_head(x)
        
        return mu, logvar

    
class ConvDecoder(BaseNetwork):
    def make_params(self, config):
        self.fc = nn.Linear(config['network.z_dim'], 256)
        
        self.feature_layers = make_transposed_cnn(input_channel=64, 
                                                  channels=[64, 64, 64], 
                                                  kernels=[4, 4, 4], 
                                                  strides=[2, 1, 1], 
                                                  paddings=[0, 0, 0], 
                                                  output_paddings=[0, 0, 0])
        
        self.x_head = nn.Linear(9216, 784)
        
    def init_params(self, config):
        ortho_init(self.fc, nonlinearity='relu', constant_bias=0.0)
        
        for layer in self.feature_layers:
            ortho_init(layer, nonlinearity='relu', constant_bias=0.0)
            
        ortho_init(self.x_head, nonlinearity='sigmoid', constant_bias=0.0)
        
    def reset(self, config, **kwargs):
        pass
    
    def forward(self, x):
        x = self.fc(x)
        
        x = x.view(-1, 64, 2, 2)
        for layer in self.feature_layers:
            x = F.relu(layer(x))
            
        x = x.flatten(start_dim=1)
        
        x = torch.sigmoid(self.x_head(x))
        
        return x
    
    
class VAE(BaseNetwork):
    def make_params(self, config):
        if config['network.type'] == 'VAE':
            self.encoder = Encoder(config=config, device=self.device)
            self.decoder = Decoder(config=config, device=self.device)
        elif config['network.type'] == 'ConvVAE':
            self.encoder = ConvEncoder(config=config, device=self.device)
            self.decoder = ConvDecoder(config=config, device=self.device)
        
    def init_params(self, config):
        pass
        
    def reset(self, config, **kwargs):
        pass
        
    def forward(self, x):
        mu, logvar = self.encoder(x)
        
        z = self.reparameterize(mu, logvar)
        
        re_x = self.decoder(z)
        
        return re_x, mu, logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        
        return mu + eps*std
    
    def vae_loss(self, re_x, x, mu, logvar, loss_type='BCE'):
        r"""Calculate `VAE loss function`_. 
        
        The VAE loss is the summation of reconstruction loss and KL loss. The KL loss
        is presented in Appendix B. 
        
        .. _VAE loss function:
            https://arxiv.org/abs/1312.6114
        
        Args:
            re_x (Tensor): reconstructed input returned from decoder
            x (Tensor): ground-truth input
            mu (Tensor): mean of the latent variable
            logvar (Tensor): log-variance of the latent variable
            loss_type (str): Type of reconstruction loss, supported ['BCE', 'MSE']
        
        Returns
        -------
        out : dict
            a dictionary of selected output such as loss, reconstruction loss and KL loss. 
        """
        assert loss_type in ['BCE', 'MSE'], f'expected either BCE or MSE, got {loss_type}'

        # shape [N, D]
        x = x.view_as(re_x)
        
        if loss_type == 'BCE':
            f = F.binary_cross_entropy
        elif loss_type == 'MSE':
            f = F.mse_loss

        re_loss = f(input=re_x, target=x, reduction='none')
        re_loss = re_loss.sum(1)
        
        KL_loss = -0.5*torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1)
        
        loss = re_loss + KL_loss
        
        out = {}
        out['loss'] = loss.mean()  # average over the batch
        out['re_loss'] = re_loss.mean()
        out['KL_loss'] = KL_loss.mean()
        
        return out
