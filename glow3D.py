import torch
import torch.nn as nn

import modules.extend_nn as nn2
import modules.couplings as couplings
import modules.mixing_channels as mix
import modules.norm as norm
import modules.functions as F2
import modules.nets as nets


class FlowStep(nn2.Sequential_Prob):
    """
    One step in the glow algorithm. Existing of normalization,
    mixing the channels and a Coupling layer.
    """
    def __init__(self, in_channels, hidden_channels):
        net = nets.net_NICE(in_channels // 2, hidden_channels)
        super().__init__(
            norm.Norm(in_channels),
            mix.InvertibleConv3d_1x1(in_channels),
            couplings.NICE(net)
        )

    def extra_repr(self):
        return "|Norm <-> Conv1x1 <-> Coupling|"

class GLOW(nn2.Sequential_Prob):
    def __init__(self, K, L, in_channels, hidden_channels):
        channels = in_channels
        sizes = ((2,2,1),) * L
        modules = []

        for i, size in enumerate(sizes):
            modules.append(nn2.Squeeze3D(size))
            channels *= (size[0] * size[1] * size[2])
            for _ in range(K):
                modules.append(FlowStep(channels, hidden_channels))

            if i != L:
                modules.append(nn2.Split(channels, channels // 2))
                channels = channels // 2

        super().__init__(*modules)
