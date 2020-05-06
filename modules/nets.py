import torch.nn as nn
import modules.extend_nn as nn2
import modules.norm as norm

def net_slim(in_channels, hidden_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, hidden_channels, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm3d(hidden_channels),
        nn.ReLU(True),
        nn.Conv3d(hidden_channels, hidden_channels, kernel_size=(1,1,3), padding=(0,0,1), bias=False),
        nn.BatchNorm3d(hidden_channels),
        nn.ReLU(True),
        nn.Conv3d(hidden_channels, in_channels, kernel_size=1, padding=0)
    )


def net_NICE(in_channels, hidden_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, hidden_channels, (1,1,3), padding=(0,0,1), bias=False),
        nn2.OnlyTensor(norm.Norm(hidden_channels)),
        nn.ReLU(True),
        nn.Conv3d(hidden_channels, in_channels, 1, bias=False)
    )
