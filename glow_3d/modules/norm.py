import torch
import torch.nn as nn

from modules.functions import num_pixels

class Norm(nn.Module):
    """
    batch normalization
    """
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
