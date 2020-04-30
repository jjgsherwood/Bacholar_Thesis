import torch
import torch.nn as nn
import torch.nn.functional as F


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
        Q = bmm_naive_cascade(self.vectors)
        return x @ Q.t()

    def inverse(self, x):
        Q = bmm_naive_cascade(self.vectors)
        return x @ Q


# Householder transformation
def _get_bmm_householder_matrices(vectors):
    N, S, _ = vectors.size()

    outer = torch.bmm(vectors, vectors.transpose(1, 2))
    inner = torch.bmm(vectors.transpose(1, 2), vectors)
    I = torch.eye(S, device=vectors.device).expand(N, -1, -1)
    hh_matrices = I - 2 * outer / (inner + 1e-16)
    return hh_matrices


def _reduce_mm(matrices):
    Q = torch.eye(matrices[0].size(0), device=matrices[0].device)
    for M in matrices:
        Q = torch.mm(Q, M)
    return Q


def bmm_naive_cascade(vectors):
    """
    Args:   
        vectors: [NumVectors, Size, 1]
    Output:
        Q: [Size, Size]
    """
    matrices = _get_bmm_householder_matrices(vectors)
    Q = _reduce_mm(matrices)
    return Q
