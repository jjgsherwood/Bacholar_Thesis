import torch
import torch.nn as nn
from torch.distributions import Normal
import modules.functions as F2


class Conv2dZeros(nn.Conv2d):
    """
    Conv2d that is init on outputting zero and is used as last layer of Real_NVP
    """
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, 3, 1, 1)
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1, 1)))
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        return output * torch.exp(self.logs * 3.0)

class OnlyTensor(nn.Module):
    """
    Makes a layer return only x instead of x, prop
    """
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)[0]

class Sequential_Prob(nn.Sequential):
    """
    nn.Sequential with input and output x, prob
    """
    def __init__(self, *args):
        for module in args:
            assert hasattr(module, 'inverse')
        super().__init__(*args)

    def forward(self, x, log_p=0):
        for module in self._modules.values():
            x, log_p = module(x, log_p)
        return x, log_p

    def inverse(self, x, log_p=0):
        for module in reversed(self._modules.values()):
            x, log_p = module.inverse(x, log_p)
        return x, log_p

class Split(nn.Module):
    """
    Splits top half features from bottom half feutures.
    So reduces the size of the networks by 2.
    """
    def __init__(self, in_features, out_features):
        assert in_features > out_features
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, log_p=0.0):
        z = x[:, self.out_features:]
        x = x[:, :self.out_features]
        z_mu = torch.zeros_like(z, device=x.device)
        z_dist = Normal(z_mu, 1)
        log_p_z = z_dist.log_prob(z).view(z.size(0), -1).sum(1)
        return x, log_p + log_p_z

    def inverse(self, x, log_p=0.0):
        size = list(x.size())
        size[1] = self.in_features - self.out_features
        z_mu = torch.zeros(*size, device=x.device)
        z_dist = Normal(z_mu, 1)
        z = z_dist.sample()
        x = torch.cat([x, z], 1)
        log_p_z = z_dist.log_prob(z).view(z.size(0), -1).sum(1)
        return x, log_p - log_p_z

    def extra_repr(self):
        return '{in_features} <~> {out_features}'.format(**self.__dict__)

class Squeeze3D(nn.Module):
    """
    Sqeezes in 3D.
    """
    def __init__(self, size=(2,2,1)):
        super().__init__()
        self.size = size

    def forward(self, x, log_p=0.0):
        return F2.squeeze3D(x, self.size), log_p

    def inverse(self, x, log_p=0.0):
        return F2.unsqueeze3D(x, self.size), log_p
