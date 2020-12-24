import torch.nn as nn
import torch
import numpy as np
from Layers import *

class VAEModel(nn.Module):
    def __init__(self, model_name, **kwargs):
        """
        PyTorch module that summarizes all components to train a VAE.
        Inputs:
            model_name - String denoting what encoder/decoder class to use.  Either 'MLP' or 'CNN'
            hidden_dims - List of hidden dimensionalities to use in the MLP layers of the encoder (decoder reversed)
            num_filters - Number of channels to use in a CNN encoder/decoder
            z_dim - Dimensionality of latent space
        """
        super().__init__()
        print(kwargs)
        for k in kwargs:
            print(k)
            # setattr(self, k, v)

        if model_name == 'MLP':
            self.encoder = MLPEncoder(self.input_dim, self.hidden_dims, self.z_dim)
            self.decoder = MLPDecoder(self.z_dim, self.hidden_dims[::-1], self.input_dim)

    def forward(self, signal):
        """
        The forward function calculates the VAE loss for a given batch of images.
        Inputs:
            imgs - Batch of tensors of shape [B,S]
        Ouptuts:
            means
            sigmas
            outputS
        """
        m, s = self.encoder(signal)
        latent = sample_reparameterize(m, s)
        return m, s, self.decoder(latent)

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Function for sampling a new batch of random images.
        Inputs:
            batch_size - Number of images to generate
        Outputs:
            x_samples - Sampled, binarized images with 0s and 1s
            x_mean - The sigmoid output of the decoder with continuous values
                     between 0 and 1 from which we obtain "x_samples"
        """
        z = torch.randn((batch_size, self.z_dim)).to(self.device)
        x_mean = torch.sigmoid(self.decoder(z))
        x_samples = torch.bernoulli(x_mean)
        return x_samples, x_mean

    @property
    def shape(self):
        return self.input_dim

    @property
    def device(self):
        """
        Property function to get the device on which the model is.
        """
        return self.device

class RNNModel(nn.Module):
    """
    Smoothing RNN
    """
    def __init__(self, lstm_num_hidden=256, lstm_num_layers=2):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(1,
                            lstm_num_hidden,
                            lstm_num_layers,
                            batch_first = True,
                            bidirectional = True)
        self.linear = nn.Linear(lstm_num_hidden * 2, 1)

    def forward(self, x, state=None):
        h, state = self.lstm(x, state)
        return self.linear(h), state

class RNN2Model(nn.Module):
    """
    Splitting RNN for Raman and AF
    """
    def __init__(self, lstm_num_hidden=256, lstm_num_layers=2):
        super(RNN2Model, self).__init__()
        self.lstm1 = nn.LSTM(1,
                            lstm_num_hidden,
                            lstm_num_layers,
                            batch_first = True,
                            bidirectional = True)
        self.lstm2 = nn.LSTM(1,
                            lstm_num_hidden,
                            lstm_num_layers,
                            batch_first = True,
                            bidirectional = True)
        self.linear1 = nn.Linear(lstm_num_hidden * 2, 1)
        self.linear2 = nn.Linear(lstm_num_hidden * 2, 1)


    def forward(self, x, state=(None, None)):
        state1, state2 = state
        h1, state1 = self.lstm1(x, state1)
        h2, state2 = self.lstm2(x, state2)
        return torch.cat((self.linear1(h1), self.linear2(h2)), 1), (state1, state2)

class RNN3Model(nn.Module):
    """
    Splitting RNN for Raman and AF plus smoothing
    """
    def __init__(self, lstm_num_hidden=256, lstm_num_layers=2):
        super(RNN3Model, self).__init__()
        self.lstm1 = nn.LSTM(1,
                            lstm_num_hidden,
                            lstm_num_layers,
                            batch_first = True,
                            bidirectional = True)
        self.lstm2 = nn.LSTM(1,
                            lstm_num_hidden,
                            lstm_num_layers,
                            batch_first = True,
                            bidirectional = True)
        self.linear1 = nn.Linear(lstm_num_hidden * 2, 1)
        self.linear2 = nn.Linear(lstm_num_hidden * 2, 1)

    def forward(self, x, state=(None, None)):
        state1, state2 = state
        h1, state1 = self.lstm1(x, state1)
        h2, state2 = self.lstm2(x, state2)
        o1, o2 = self.linear1(h1), self.linear2(h2)
        return torch.cat((o1, o2, o1 + o2), 1), (state1, state2)

class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))
