
import torch, math

def precompute_freqs(dim, seq, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq, dtype=torch.float32)
    return torch.polar(torch.ones_like(freqs)[None].expand(seq, -1), torch.outer(t, freqs))

def apply_rope(x, freqs):
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * freqs[:x.shape[-2]].to(x.device)).flatten(-2).type_as(x)
