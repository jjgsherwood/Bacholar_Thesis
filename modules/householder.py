import torch
import torch.nn as nn    
    
    
class HH(nn.Module):
    def __init__(self, size, num_vectors=None):
        super().__init__()
        self.size = size
        self.num_vectors = num_vectors or 2 * (self.size // 2 + 1)
        self.vectors = nn.Parameter(torch.Tensor(self.num_vectors, self.size, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.vectors, -1, 1)
        self.vectors.data.copy_(self.vectors / self.vectors.norm(dim=1, keepdim=True))

    def forward(self, x):
        Q = self.calculate_Q()
        return x @ Q.t()

    def inverse(self, x):
        Q = self.calculate_Q()
        return x @ Q

    def calculate_Q(self):
        N, S, _ = self.vectors.size()

        outer = torch.bmm(self.vectors, self.vectors.transpose(1, 2))
        inner = torch.bmm(self.vectors.transpose(1, 2), self.vectors)
        I = torch.eye(S, device=self.vectors.device).expand(N, -1, -1)
        matrices = I - 2 * outer / (inner + 1e-16)
        Q = matrices[0]
        for M in matrices[1:]:
            Q = torch.mm(Q, M)
        return Q
    
