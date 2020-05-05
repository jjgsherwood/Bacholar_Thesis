import torch
import torch.nn as nn


class Conv2dZeros(nn.Conv2d):
    """

    """
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, 3, 1, 1)
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1, 1)))
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        return output * torch.exp(self.logs * 3.0)
