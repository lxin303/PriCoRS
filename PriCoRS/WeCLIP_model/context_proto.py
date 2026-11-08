import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Dict, List, Tuple, Optional

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

class FIFOQueue:
    def __init__(self, maxlen: int):
        self.q = deque(maxlen=maxlen)
    def push(self, x: torch.Tensor):
        self.q.append(x.detach().cpu())
    def as_tensor(self) -> Optional[torch.Tensor]:
        if len(self.q) == 0:
            return None
        return torch.stack(list(self.q), dim=0)  # [M, Dp]
    def __len__(self): return len(self.q)

class OnlineKMeans:
    def __init__(self, num_clusters: int, dim: int, momentum: float = 0.5):
        self.K = num_clusters
        self.dim = dim
        self.m = momentum
        self.inited = False
        self.centroids = None  # [K, dim]

    def _init_from(self, X: torch.Tensor):
        # X: [M, dim] (cpu)
        M = X.shape[0]
        if M < self.K:
            # pad with random rows from X (with replacement)
            idx = torch.randint(0, M, (self.K - M,))
            X = torch.cat([X, X[idx]], dim=0)
        # simple kmeans++-ish: random subset
        ids = torch.randperm(X.shape[0])[:self.K]
        self.centroids = X[ids].clone()
        self.inited = True

    def partial_fit(self, X: torch.Tensor):
        X = X.detach().cpu()
        if not self.inited:
            self._init_from(X)
        for x in X:
            sims = torch.cosine_similarity(self.centroids, x[None, :], dim=1)  # [K]
            idx = torch.argmax(sims).item()
            self.centroids[idx] = self.m * self.centroids[idx] + (1 - self.m) * x

    def get_prototypes(self) -> Optional[torch.Tensor]:
        if not self.inited or self.centroids is None:
            return None
        return self.centroids.clone()  # [K, dim] (cpu)

class ContextPrototypeAware(nn.Module):
    """
    - w_i (softmax/temperature)
    - Top-K 
    - PACAM
    """
    def __init__(self, dim: int, top_k: int = 5, temperature: float = 0.07,
                 weighted: bool = False, learnable_maps: bool = False):
        super().__init__()
        self.dim = dim
        self.top_k = top_k
        self.temperature = temperature
        self.weighted = weighted
        if learnable_maps:
            self.l1 = nn.Linear(dim, dim, bias=False)
            self.l2 = nn.Linear(dim, dim, bias=False)
            nn.init.eye_(self.l1.weight); nn.init.eye_(self.l2.weight)
        else:
            self.l1 = nn.Identity()
            self.l2 = nn.Identity()

    @torch.no_grad()
    def compute_delta(self, P_I_batch: torch.Tensor, P_cand: torch.Tensor) -> torch.Tensor:
        # δ_n = mean(pi) - mean(PI)
        return P_cand.mean(dim=0) - P_I_batch.mean(dim=0)

    def _positiveness_topk(self, P_I: torch.Tensor, P_cand: torch.Tensor):
        # P_I: [Dp] (cuda), P_cand: [Np, Dp] (cpu or cuda)
        if P_cand.device != P_I.device:
            P_cand = P_cand.to(P_I.device)
        q = l2_normalize(self.l1(P_I), dim=0)              # [Dp]
        keys = l2_normalize(self.l2(P_cand), dim=1)        # [Np, Dp]
        sims = (keys @ q)                                  # [Np]
        w = F.softmax(sims / self.temperature, dim=0)      # [Np]
        K = min(self.top_k, w.numel())
        w_top, idx = torch.topk(w, K, dim=0, largest=True, sorted=True)
        P_top = P_cand.index_select(0, idx)
        return P_top, w_top, idx

    def pacam(self, z: torch.Tensor, topP: torch.Tensor, w_top: Optional[torch.Tensor]):
        # z: [Dp, H, W]  topP: [K, Dp]
        Dp, H, W = z.shape
        z_n = l2_normalize(z, dim=0)                       # [Dp, H, W]
        Pn = l2_normalize(topP, dim=1)                     # [K, Dp]
        sims = Pn @ z_n.reshape(Dp, H*W)                   # [K, H*W]
        sims = sims.view(-1, H, W)                         # [K, H, W]
        if self.weighted and w_top is not None:
            w = w_top / (w_top.sum() + 1e-6)
            pacam = (w[:, None, None] * sims).sum(0)       # [H, W]
        else:
            pacam = sims.mean(0)
        return F.relu(pacam)  