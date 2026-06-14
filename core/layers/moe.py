
import torch, torch.nn as nn

class Expert(nn.Module):
    def __init__(self, dim, inter):
        super().__init__()
        self.w1 = nn.Linear(dim, inter, bias=False)
        self.w2 = nn.Linear(inter, dim, bias=False)
    def forward(self, x): return self.w2(torch.nn.functional.silu(self.w1(x)))

class MoELayer(nn.Module):
    def __init__(self, dim, inter, n_experts=8, top_k=2):
        super().__init__()
        self.n_experts, self.top_k = n_experts, top_k
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(dim, inter) for _ in range(n_experts)])
    def forward(self, x):
        B, L, D = x.shape
        logits = self.gate(x.view(-1, D))
        weights, indices = torch.topk(torch.softmax(logits, dim=-1), self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(x.view(-1, D))
        for i in range(self.n_experts):
            mask = (indices == i).any(dim=-1)
            if mask.any():
                exp_out = self.experts[i](x.view(-1, D)[mask])
                for k in range(self.top_k):
                    km = (indices[:, k] == i) & mask
                    if km.any(): out[km] += weights[km, k].unsqueeze(-1) * exp_out[:km.sum()]
        aux_loss = logits.softmax(-1).float().mean(0).pow(2).sum() * self.n_experts
        return out.view(B, L, D), aux_loss * 0.01
