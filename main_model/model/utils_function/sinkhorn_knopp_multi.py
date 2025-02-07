import torch
import torch.nn.functional as F
from icecream import ic
def sinkhorn_knopp_uniform(similarity_matrix, epsilon=1, num_iters=5, detach = True):
    if detach:
        similarity_matrix = similarity_matrix.detach()
    # Ensure the input is a PyTorch tensor
    S = torch.tensor(similarity_matrix, dtype=torch.float).to(device=similarity_matrix.device)

    # Assuming uniform distributions for mu and nu based on the size of S
    mu = (torch.ones(S.size(0)) / S.size(0)).to(device=similarity_matrix.device)
    nu = (torch.ones(S.size(1)) / S.size(1)).to(device=similarity_matrix.device)

    # Initialize kappa (k1 and k2 in the formula)
    k1 = torch.ones_like(mu).to(device=similarity_matrix.device)
    k2 = torch.ones_like(nu).to(device=similarity_matrix.device)

    # Sinkhorn iterations
    for _ in range(num_iters):
        k1 = mu / (torch.exp(S / epsilon) @ k2)
        k2 = nu / (torch.exp(S.T / epsilon) @ k1)

    # Construct the optimal transport matrix Q for E-step
    Q = torch.diag(k1) @ torch.exp(S / epsilon) @ torch.diag(k2)

    return Q

