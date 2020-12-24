import torch.nn as nn
import torch
import numpy as np


class MLPEncoder(nn.Module):
    def __init__(self, input_dim=[784], n_hidden=[512], z_dim=20):
        """
        Encoder with an MLP network and ReLU activations (except the output layer).

        Inputs:
            input_dim - Number of input neurons/pixels. For MNIST, 28*28=784
            hidden_dims - List of dimensionalities of the hidden layers in the network.
                          The NN should have the same number of hidden layers as the length of the list.
            z_dim - Dimensionality of latent vector.
        """
        super().__init__()

        # For an intial architecture, you can use a sequence of linear layers and ReLU activations.
        # Feel free to experiment with the architecture yourself, but the one specified here is
        # sufficient for the assignment.

        # COPIED FROM ASSIGNMENT 1
        self.full = []
        self.n_inputs = prev = np.prod(input_dim)

        for next in n_hidden:
            self.full.append(nn.Linear(prev, next))
            prev = next
            self.full.append(nn.ReLU(True))

        self.full.append(nn.Linear(prev, z_dim*2))
        self.full = nn.Sequential(*self.full)

    def forward(self, x):
        """
        Inputs:
            x - Input batch with tensors of shape [B,C,H,W] and range 0 to 1.
        Outputs:
            mean - Tensor of shape [B,z_dim] representing the predicted mean of the latent distributions.
            log_std - Tensor of shape [B,z_dim] representing the predicted log standard deviation
                      of the latent distributions.
        """

        # Remark: Make sure to understand why we are predicting the log_std and not std
        return self.full(x.reshape(-1, self.n_inputs)).chunk(2, 1)

class MLPDecoder(nn.Module):
    def __init__(self, z_dim=20, n_hidden=[512], output_shape=[1,28,28]):
        """
        Decoder with an MLP network.
        Inputs:
            z_dim - Dimensionality of latent vector (input to the network).
            hidden_dims - List of dimensionalities of the hidden layers in the network.
                          The NN should have the same number of hidden layers as the length of the list.
            output_shape - Shape of output image. The number of output neurons of the NN must be
                           the product of the shape elements.
        """
        super().__init__()
        self.output_shape = output_shape
        self.full = []
        prev = z_dim

        for next in n_hidden:
            self.full.append(nn.Linear(prev, next))
            prev = next
            self.full.append(nn.ReLU(True))

        self.full.append(nn.Linear(prev, np.prod(output_shape)))
        self.full = nn.Sequential(*self.full)
        # For an intial architecture, you can use a sequence of linear layers and ReLU activations.
        # Feel free to experiment with the architecture yourself, but the one specified here is
        # sufficient for the assignment.

    def forward(self, z):
        """
        Inputs:
            z - Latent vector of shape [B,z_dim]
        Outputs:
            x - Prediction of the reconstructed image based on z.
                This should be a logit output *without* a sigmoid applied on it.
                Shape: [B,output_shape[0],output_shape[1],output_shape[2]]
        """
        return self.full(z).reshape(-1, *self.output_shape)

    @property
    def device(self):
        """
        Property function to get the device on which the decoder is.
        Might be helpful in other functions.
        """
        return next(self.parameters()).device
