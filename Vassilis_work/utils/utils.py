import torch
import torch.nn as nn
import numpy as np

def sample_reparameterize(mean, std):
    """
    Perform the reparameterization trick to sample from a distribution with the given mean and std
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions
        std - Tensor of arbitrary shape with strictly positive values. Denotes the standard deviation
              of the distribution
    Outputs:
        z - A sample of the distributions, with gradient support for both mean and std.
            The tensor should have the same shape as the mean and std input tensors.
    """
    z = torch.normal(0, 1, mean.shape).to(mean.device)
    return z * torch.exp(std) + mean


def KLD(mean, log_std):
    """
    Calculates the Kullback-Leibler divergence of given distributions to unit Gaussians over the last dimension.
    See Section 1.3 for the formula.
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions.
        log_std - Tensor of arbitrary shape and range, denoting the log standard deviation of the distributions.
    Outputs:
        KLD - Tensor with one less dimension than mean and log_std (summed over last dimension).
              The values represent the Kullback-Leibler divergence to unit Gaussians.
    """

    return (torch.exp(2*log_std) + mean**2 - 1 - 2*log_std).sum(-1) / 2
