
import torch, torch.nn as nn

class SwiGLU(nn.Module):
    def __init__(self, dim: int, inter: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter, bias=False)
        self.w2 = nn.Linear(inter, dim, bias=False)
        self.w3 = nn.Linear(dim, inter, bias=False)
    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))

class FFN(nn.Module):
    def __init__(self, dim: int, inter: int):
        super().__init__()
        self.gate = nn.Linear(dim, inter, bias=False)
        self.up = nn.Linear(dim, inter, bias=False)
        self.down = nn.Linear(inter, dim, bias=False)
    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))
