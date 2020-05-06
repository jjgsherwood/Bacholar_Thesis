import torch
import torch.nn as nn
import torch.nn.functional as F

from flows import Squeeze, iSequential, Norm, InvertibleConv2d_1x1, Split


class OnlyTensor(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)[0]


class Conv2dZeros(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, 3, 1, 1)
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1, 1)))
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        return output * torch.exp(self.logs * 3.0)


class Coupling(nn.Module):

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.scale_shift = nn.Sequential(
            nn.Conv2d(in_channels // 2, hidden_channels, 3, padding=1, bias=False),
            OnlyTensor(Norm(hidden_channels)),
            nn.ReLU(False),
            nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False),
            OnlyTensor(Norm(hidden_channels)),
            nn.ReLU(False),
            Conv2dZeros(hidden_channels, in_channels)
        )

    def forward(self, x, log_p=0.0):
        C = x.size(1) // 2
        x1, x2 = x.split(C, dim=1)
        x1 = x1.contiguous()
        x2 = x2.contiguous()

        scale, shift = self.scale_shift(x2).split(C, dim=1)
        scale = torch.sigmoid(scale + 2.0)
        x1 = (x1 + shift) * scale

        log_det = torch.log(scale).mean(0).sum()
        return torch.cat([x1, x2], dim=1), log_p + log_det

    def inverse(self, x, log_p=0.0):
        C = x.size(1) // 2
        x1, x2 = x.split(C, dim=1)
        x1 = x1.contiguous()
        x2 = x2.contiguous()

        scale, shift = self.scale_shift(x2).split(C, dim=1)
        scale = torch.sigmoid(scale + 2.0)
        x1 = x1 / scale - shift
        log_det = torch.log(scale).mean(0).sum()
        return torch.cat([x1, x2], dim=1), log_p - log_det


class FlowStep(iSequential):

    def __init__(self, in_channels, hidden_channels):
        super().__init__(
            Norm(in_channels),
            InvertibleConv2d_1x1(in_channels),
            Coupling(in_channels, hidden_channels)
        )

    def extra_repr(self):
        return "|Norm <-> Conv1x1 <-> Coupling|"


class GLOW(iSequential):

    def __init__(self, K, L, in_channels, hidden_channels):
        channels = in_channels
        modules = []

        for _ in range(L-1):
            modules.append(Squeeze())
            channels *= 4
            for _ in range(K):
                modules.append(FlowStep(channels, hidden_channels))

            modules.append(Split(channels, channels // 2))
            channels = channels // 2

        modules.append(Squeeze())
        channels *= 4
        for _ in range(K):
            modules.append(FlowStep(channels, hidden_channels))

        super().__init__(*modules)
