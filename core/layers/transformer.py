
import torch, torch.nn as nn
from core.layers.attention import Attention
from core.layers.moe import MoELayer
from core.layers.norm import RMSNorm
from core.layers.ffn import FFN

class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = Attention(cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len)
        self.norm1 = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.ffn = MoELayer(cfg.hidden_size, cfg.intermediate_size, cfg.num_experts, cfg.experts_per_tok) if cfg.num_experts > 1 else FFN(cfg.hidden_size, cfg.intermediate_size)
        self.norm2 = RMSNorm(cfg.hidden_size, cfg.norm_eps)
    def forward(self, x, mask=None, pos=0):
        x = x + self.attn(self.norm1(x), mask, pos)
        out, aux = self.ffn(self.norm2(x)) if isinstance(self.ffn, MoELayer) else (self.ffn(self.norm2(x)), torch.tensor(0.0))
        return x + out, aux

class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings: self.head.weight = self.embed.weight

    def forward(self, ids, labels=None, mask=None):
        x = self.embed(ids)
        total_aux = torch.tensor(0.0, device=ids.device)
        for i, layer in enumerate(self.layers):
            x, aux = layer(x, mask, pos=0)
            total_aux = total_aux + aux
        logits = self.head(self.norm(x))
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100)
            loss = loss + total_aux / len(self.layers)
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, max_new=256, temp=0.7, top_k=50, top_p=0.9):
        self.eval()
        for _ in range(max_new):
            logits, _ = self.forward(ids[:, -self.cfg.max_seq_len:])
            next_logits = logits[:, -1, :] / max(temp, 1e-7)
            if top_k > 0:
                thresh = torch.topk(next_logits, top_k)[0][:, -1:]
                next_logits[next_logits < thresh] = float('-inf')
            if top_p < 1.0:
                sorted_l, sorted_i = torch.sort(next_logits, descending=True)
                cum = torch.cumsum(torch.softmax(sorted_l, dim=-1), dim=-1)
                remove = cum > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_l[remove] = float('-inf')
                next_logits.scatter_(1, sorted_i, sorted_l)
            next_tok = torch.multinomial(torch.softmax(next_logits, dim=-1), 1)
            ids = torch.cat([ids, next_tok], dim=1)
        return ids
