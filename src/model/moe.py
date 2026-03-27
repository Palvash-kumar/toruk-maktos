import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# EXPERT NETWORK
# -----------------------------

class Expert(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):

        return self.net(x)


# -----------------------------
# ROUTER
# -----------------------------

class Router(nn.Module):

    def __init__(
        self,
        dim,
        num_experts
    ):

        super().__init__()

        self.linear = nn.Linear(
            dim,
            num_experts
        )

    def forward(self, x):

        scores = self.linear(x)

        return scores


# -----------------------------
# SPECIFIC MoE
# -----------------------------

class SpecificMoE(nn.Module):

    def __init__(
        self,
        dim,
        num_experts=4,
        top_k=2
    ):

        super().__init__()

        self.experts = nn.ModuleList(
            [
                Expert(dim)
                for _ in range(num_experts)
            ]
        )

        self.router = Router(
            dim,
            num_experts
        )

        self.top_k = top_k

    def forward(self, x):

        # x shape:
        # (batch, seq, dim)

        scores = self.router(x)

        topk = scores.topk(
            self.top_k,
            dim=-1
        )

        topk_indices = topk.indices

        output = torch.zeros_like(x)

        batch_size = x.shape[0]

        for b in range(batch_size):

            for i in range(self.top_k):

                expert_index = int(
                    topk_indices[b, 0, i]
                )

                expert = self.experts[
                    expert_index
                ]

                # FIX: correct tensor input
                output[b] += (
                    expert(x[b])
                    / self.top_k
                )

        return output


# -----------------------------
# SHARED MoE
# -----------------------------

class SharedMoE(nn.Module):

    def __init__(
        self,
        dim,
        num_experts=2
    ):

        super().__init__()

        self.experts = nn.ModuleList(
            [
                Expert(dim)
                for _ in range(num_experts)
            ]
        )

    def forward(self, x):

        output = torch.zeros_like(x)

        for expert in self.experts:

            output += expert(x)

        output = output / len(self.experts)

        return output


# -----------------------------
# DOMAIN-DECOUPLED MoE
# -----------------------------

class SSMoE(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.specific = SpecificMoE(dim)

        self.shared = SharedMoE(dim)

    def forward(self, x):

        return (
            self.specific(x)
            + self.shared(x)
        )