import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import sys
sys.path.append('../')




class SpikingActor(nn.Module):
    def __init__(self, obs_dim, act_dim, en_pop_dim, de_pop_dim, hidden_sizes,
                 mean_range, std, spike_ts, device):
        """
        :param obs_dim: observation dimension
        :param act_dim: action dimension
        :param en_pop_dim: encoder population dimension
        :param de_pop_dim: decoder population dimension
        :param hidden_sizes: list of hidden layer sizes
        :param mean_range: mean range for encoder
        :param std: std for encoder
        :param spike_ts: spike timesteps
        :param device: device
        """
        super().__init__()
        self.encoder = PopSpikeEncoderRegularSpike(obs_dim, en_pop_dim, spike_ts, mean_range, std, device)
        self.snn = SpikeMLP(obs_dim*en_pop_dim, act_dim*de_pop_dim, hidden_sizes, spike_ts, device)
        self.decoder = PopSpikeDecoder(act_dim, de_pop_dim)
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def forward(self, obs, batch_size):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: action scale with action limit
        """
        in_pop_spikes = self.encoder(obs, batch_size)
        out_pop_activity = self.snn(in_pop_spikes, batch_size)
        mu = self.decoder(out_pop_activity)
        # std = torch.exp(self.log_std)
        return mu, None