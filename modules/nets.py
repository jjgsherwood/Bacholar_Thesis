import torch.nn as nn
import modules.extend_nn as nn2
import modules.norm as norm

def net_mnist_Real_NVP(in_channels, hidden_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, hidden_channels, kernel_size=(3,3,1), padding=(1,1,0), bias=False),
        nn2.OnlyTensor(norm.Norm(hidden_channels)),
        nn.ReLU(True),
        nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1, padding=0, bias=False),
        nn2.OnlyTensor(norm.Norm(hidden_channels)),
        nn.ReLU(True),
    )

def net_mnist_NICE(in_channels, hidden_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, hidden_channels, (3,3,1), padding=(1,1,0), bias=False),
        nn2.OnlyTensor(norm.Norm(hidden_channels)),
        nn.ReLU(True),
        nn.Conv3d(hidden_channels, in_channels, kernel_size=1, padding=0, bias=False),
    )

def net_liver_NICE(in_channels, hidden_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, hidden_channels, (1,1,5), padding=(0,0,2), bias=False),
        nn2.OnlyTensor(norm.Norm(hidden_channels)),
        nn.ReLU(True),
        nn.Conv3d(hidden_channels, hidden_channels, (1,1,3), padding=(0,0,1), bias=False),
        nn2.OnlyTensor(norm.Norm(hidden_channels)),
        nn.ReLU(True),
        nn.Conv3d(hidden_channels, in_channels, kernel_size=1, padding=0, bias=False)
    )

def net_liver_Real_NVP(in_channels, hidden_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, hidden_channels, (1,1,5), padding=(0,0,2), bias=False),
        nn2.OnlyTensor(norm.Norm(hidden_channels)),
        nn.ReLU(True),
        nn.Conv3d(hidden_channels, hidden_channels, (1,1,3), padding=(0,0,1), bias=False),
        nn2.OnlyTensor(norm.Norm(hidden_channels)),
        nn.ReLU(True),
        nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1, padding=0, bias=False),
        nn2.OnlyTensor(norm.Norm(hidden_channels)),
        nn.ReLU(True),
    )

def net_liver_large(in_channels, hidden_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, hidden_channels, (1,1,7), padding=(0,0,3), bias=False),
        nn2.OnlyTensor(norm.Norm(hidden_channels)),
        nn.ReLU(True),
        nn.Conv3d(hidden_channels, hidden_channels, (1,1,5), padding=(0,0,2), bias=False),
        nn2.OnlyTensor(norm.Norm(hidden_channels)),
        nn.ReLU(True),
        nn.Conv3d(hidden_channels, in_channels, kernel_size=1, padding=0, bias=False)
    )
