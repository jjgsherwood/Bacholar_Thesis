import torch
import torch.nn as nn 


class NICE(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        
    def forward(self, x):
        x1, x2 = x.split(x.size(1) // 2, 1)
        x2 = x2 + self.net(x1)
        return torch.cat([x2, x1], 1)
    
    def inverse(self, x):
        x2, x1 = x.split(x.size(1) // 2, 1)
        x2 = x2 - self.net(x1)
        return torch.cat([x1, x2], 1)
    
