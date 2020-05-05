import torch.nn as nn

class Sequential_Prob(nn.Sequential):
    """
    nn.Sequential with input and output x, prob
    """
    def __init__(self, *args):
        for module in args:
            assert hasattr(module, 'inverse')
        super(iSequential, self).__init__(*args)

    def forward(self, x, log_p=0):
        for module in self._modules.values():
            x, log_p = module(x, log_p)
        return x, log_p

    def inverse(self, x, log_p=0):
        for module in reversed(self._modules.values()):
            x, log_p = module.inverse(x, log_p)
        return x, log_p
