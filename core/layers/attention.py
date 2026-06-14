
import math, torch, torch.nn as nn, torch.nn.functional as F
from core.layers.rope import precompute_freqs, apply_rope

class Attention(nn.Module):
    def __init__(self, dim, nheads, nkvh, head_dim, max_seq=4096):
        super().__init__()
        self.nheads, self.nkvh, self.head_dim = nheads, nkvh, head_dim
        self.qkv = nn.Linear(dim, (nheads + 2*nkvh) * head_dim, bias=False)
        self.out = nn.Linear(nheads * head_dim, dim, bias=False)
        self.register_buffer("freqs", precompute_freqs(head_dim, max_seq * 2), persistent=False)

    def forward(self, x, mask=None, pos=0):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, -1, self.head_dim)
        q, k, v = qkv.split([self.nheads, self.nkvh, self.nkvh], dim=2)
        q = apply_rope(q, self.freqs[pos:pos+L])
        k = apply_rope(k, self.freqs[pos:pos+L])
        k = k.repeat_interleave(self.nheads // self.nkvh, dim=2)
        v = v.repeat_interleave(self.nheads // self.nkvh, dim=2)
        try:
            out = F.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), is_causal=mask is None)
        except:
            w = (q @ k.transpose(-2,-1)) / math.sqrt(self.head_dim)
            if mask is not None: w = w + mask
            out = (F.softmax(w, dim=-1) @ v).transpose(1,2)
        return self.out(out.reshape(B, L, -1))
