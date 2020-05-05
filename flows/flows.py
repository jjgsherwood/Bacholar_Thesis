import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def num_pixels(tensor):
    num_elements = tensor.nelement()
    B, C = tensor.size(0), tensor.size(1)
    return num_elements / B / C


class iSequential(nn.Sequential):

    def __init__(self, *args):
        for module in args:
            assert hasattr(module, 'inverse')
        super(iSequential, self).__init__(*args)

    def forward(self, x, log_p=0.0):
        for module in self._modules.values():
            x, log_p = module(x, log_p=log_p)
        return x, log_p

    def inverse(self, x, log_p=0.0):
        for module in reversed(self._modules.values()):
            x, log_p = module.inverse(x, log_p=log_p)
        return x, log_p


class Split(nn.Module):

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


class InvertibleConv2d_1x1(nn.Module):
    '''
    Diederik P. Kingma, Prafulla Dhariwal
    "Glow: Generative Flow with Invertible 1x1 Convolutions"
    https://arxiv.org/pdf/1807.03039.pdf
    '''

    def __init__(self, features):
        super().__init__()
        self.features = features
        self.W = nn.Parameter(torch.Tensor(features, features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.W)

    def forward(self, x, log_p=0.0):
        log_det = torch.log(torch.abs(torch.det(self.W.double()))).float() * num_pixels(x)
        W = self.W.unsqueeze(-1).unsqueeze(-1)
        return F.conv2d(x, W), log_p + log_det

    def inverse(self, x, log_p=0.0):
        log_det = torch.log(torch.abs(torch.det(self.W.double()))).float() * num_pixels(x)
        W = torch.inverse(self.W)
        W = W.unsqueeze(-1).unsqueeze(-1)
        return F.conv2d(x, W), log_p - log_det


class Norm(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.logs = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.requires_init = nn.Parameter(torch.ByteTensor(1), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.logs.data.zero_()
        self.bias.data.zero_()
        self.requires_init.data.fill_(True)

    def init_data_dependent(self, x):
        with torch.no_grad():
            x_ = x.transpose(0, 1).contiguous().view(self.num_features, -1)
            mean = x_.mean(1)
            var = x_.var(1)
            logs = torch.log(1.0 / (torch.sqrt(var) + 1e-6))
            self.logs.data.copy_(logs.data)
            self.bias.data.copy_(mean.data)

    def forward(self, x, log_p=0.0):
        assert x.size(1) == self.num_features
        if self.requires_init:
            self.requires_init.data.fill_(False)
            self.init_data_dependent(x)

        size = [1] * x.ndimension()
        size[1] = self.num_features
        x = (x - self.bias.view(*size)) * torch.exp(self.logs.view(*size))
        log_det = self.logs.sum() * num_pixels(x)
        return x, log_p + log_det

    def inverse(self, x, log_p=0.0):
        size = [1] * x.ndimension()
        size[1] = self.num_features
        x = x * torch.exp(-self.logs.view(*size)) + self.bias.view(*size)
        log_det = self.logs.sum() * num_pixels(x)
        return x, log_p - log_det

    def extra_repr(self):
        return '{}, requires_init={}'.format(self.num_features, bool(self.requires_init.item()))


class Squeeze(nn.Module):

    def __init__(self, size=2):
        super().__init__()
        self.size = size

    def forward(self, x, log_p=0.0):
        return squeeze(x, self.size), log_p

    def inverse(self, x, log_p=0.0):
        return unsqueeze(x, self.size), log_p


def unsqueeze(input, upscale_factor=2):
    '''
    [:, C*r^2, H, W] -> [:, C, H*r, W*r]
    '''
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels = in_channels // (upscale_factor**2)

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor

    input_view = input.contiguous().view(batch_size, out_channels, upscale_factor,
                                         upscale_factor, in_height, in_width)

    output = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    return output.view(batch_size, out_channels, out_height, out_width)


def squeeze(input, downscale_factor=2):
    '''
    [:, C, H*r, W*r] -> [:, C*r^2, H, W]
    '''
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels = in_channels * (downscale_factor**2)

    out_height = in_height // downscale_factor
    out_width = in_width // downscale_factor

    input_view = input.contiguous().view(
        batch_size, in_channels, out_height, downscale_factor, out_width, downscale_factor
    )

    output = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return output.view(batch_size, out_channels, out_height, out_width)
