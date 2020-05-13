import torch
import torch.nn as nn
import modules.extend_nn as nn2
import modules.functions as F2

class NICE(nn.Module):
    """
    net is a nn.Sequential with as in_channels and out_channels half the
    channels of x.
    """
    def __init__(self, net):
        super().__init__()
        self.shift = net

    def forward(self, x, log_p=0.0):
        C = x.size(1) // 2
        x1, x2 = x.split(C, 1)
        x1 = x1.contiguous()
        x2 = x2.contiguous()

        x2 = x2 + self.shift(x1)
        return torch.cat([x2, x1], 1), log_p

    def inverse(self, x, log_p=0.0):
        C = x.size(1) // 2
        x2, x1 = x.split(C, 1)
        x1 = x1.contiguous()
        x2 = x2.contiguous()

        x2 = x2 - self.shift(x1)
        return torch.cat([x1, x2], 1), log_p

class Real_NVP(nn.Module):
    """
    net is a nn.Sequential with as in_channels half the
    channels of x, out_channels can be anny number (hidden layer).
    A last layer is added to multiply the number of channels with 2 to
    produce scale and shift and to make sure the scale is init on 0.88.
    sigmoid(0 + 2.0) = 0.88
    """
    def __init__(self, net):
        super().__init__()
        in_channels, hidden_channels = F2.get_in_out_channels(net)
        self.scale_shift = nn.Sequential(*net,
            nn2.Conv3dZeros(hidden_channels, in_channels * 2))

    def forward(self, x, log_p=0.0):
        C = x.size(1) // 2
        x1, x2 = x.split(C, 1)
        x1 = x1.contiguous()
        x2 = x2.contiguous()

        scale, shift = self.scale_shift(x2).split(C, dim=1)
        scale = torch.sigmoid(scale + 2.0)
        x1 = (x1 + shift) * scale

        log_det = torch.log(scale).sum()
        return torch.cat([x1, x2], dim=1), log_p + log_det

    def inverse(self, x, log_p=0.0):
        C = x.size(1) // 2
        x1, x2 = x.split(C, dim=1)
        x1 = x1.contiguous()
        x2 = x2.contiguous()

        scale, shift = self.scale_shift(x2).split(C, dim=1)
        scale = torch.sigmoid(scale + 2.0)
        x1 = x1 / scale - shift
        log_det = torch.log(scale).sum()
        return torch.cat([x1, x2], dim=1), log_p - log_det

class NICE_oneven(nn.Module):
    """
    net is a nn.Sequential with as in_channels and out_channels half the
    channels of x.
    """
    def __init__(self, net):
        super().__init__()
        self.shift = net

    def forward(self, x, log_p=0.0):
        C = int(x.size(1) / 2 + 0.5)
        x1, x2 = x.split(C, 1)
        x1 = x1.contiguous()
        x2 = x2.contiguous()

        x2 = x2 + self.shift(x1)
        return torch.cat([x2, x1], 1), log_p

    def inverse(self, x, log_p=0.0):
        C = x.size(1) // 2
        x2, x1 = x.split(C, 1)
        x1 = x1.contiguous()
        x2 = x2.contiguous()

        x2 = x2 - self.shift(x1)
        return torch.cat([x1, x2], 1), log_p
