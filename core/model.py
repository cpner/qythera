import numpy as np
import math
from core.nn.module import Module, Parameter
from core.nn.embedding import Embedding
from core.nn.norm import RMSNorm
from core.nn.attention import MultiHeadAttention
from core.nn.moe import MoELayer
from core.nn.ffn import FeedForward
from core.autodiff.tensor import Tensor


class VaelonConfig:
    """Model configuration."""
    def __init__(self, vocab_size=32000, hidden_size=1024, num_layers=12,
                 num_heads=8, num_kv_heads=2, head_dim=128, intermediate_size=2816,
                 max_seq_len=2048, num_experts=4, experts_per_tok=2,
                 dropout=0.0, norm_eps=1e-6, rope_theta=10000.0):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        self.num_experts = num_experts
        self.experts_per_tok = experts_per_tok
        self.dropout = dropout
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta

    @classmethod
    def small(cls):
        return cls(hidden_size=512, num_layers=6, num_heads=4, num_kv_heads=2,
                   head_dim=128, intermediate_size=1376, num_experts=2)
    @classmethod
    def medium(cls):
        return cls(hidden_size=1024, num_layers=12, num_heads=8, num_kv_heads=2,
                   head_dim=128, intermediate_size=2816, num_experts=4)
    @classmethod
    def large(cls):
        return cls(hidden_size=2048, num_layers=24, num_heads=16, num_kv_heads=4,
                   head_dim=128, intermediate_size=5504, num_experts=8)


class VaelonDecoderLayer(Module):
    """Single decoder layer: attention + MoE FFN with pre-norm."""
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.attn = MultiHeadAttention(
            config.hidden_size, config.num_heads, config.num_kv_heads,
            config.head_dim, config.max_seq_len, config.dropout
        )
        self.ffn_norm = RMSNorm(config.hidden_size, config.norm_eps)
        if config.num_experts > 1:
            self.ffn = MoELayer(config.hidden_size, config.intermediate_size,
                                config.num_experts, config.experts_per_tok)
        else:
            self.ffn = FeedForward(config.hidden_size, config.intermediate_size)

    def forward(self, x, mask=None, kv_cache=None, position=0):
        h = self.attn_norm(x)
        attn_out, new_cache = self.attn(h, mask, kv_cache, position)
        x = x + attn_out

        h = self.ffn_norm(x)
        if isinstance(self.ffn, MoELayer):
            ffn_out, aux_loss = self.ffn(h)
        else:
            ffn_out = self.ffn(h)
            aux_loss = 0.0
        x = x + ffn_out
        return x, new_cache, aux_loss


class VaelonModel(Module):
    """Vaelon Transformer with Mixture of Experts.
    
    Architecture:
    - Token Embeddings
    - N Decoder Layers (Attention + MoE FFN)
    - Final RMSNorm
    - LM Head (vocab projection)
    
    Supports:
    - Forward pass with labels for training
    - Autoregressive generation with KV-cache
    - Load balancing loss for MoE
    """
    
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = VaelonConfig()
        self.config = config

        self.embed = Embedding(config.vocab_size, config.hidden_size)
        self.layers = [VaelonDecoderLayer(config) for _ in range(config.num_layers)]
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.head = Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie embeddings
        self.head.weight = self.embed.weight

    def forward(self, ids, labels=None, mask=None):
        """Forward pass.
        
        Args:
            ids: Token indices (B, L)
            labels: Target indices for loss computation (B, L)
            mask: Attention mask
        
        Returns:
            logits, loss (if labels provided), aux_loss (MoE)
        """
        if isinstance(ids, np.ndarray):
            ids = Tensor(ids)
        if ids.data.dtype != np.int32:
            ids = Tensor(ids.data.astype(np.int32))

        x = self.embed(ids)
        total_aux = 0.0

        for i, layer in enumerate(self.layers):
            x, _, aux = layer(x, mask, position=0)
            total_aux += aux

        x = self.norm(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            if isinstance(labels, np.ndarray):
                labels = Tensor(labels.astype(np.int32))
            shift_logits = Tensor(logits.data[:, :-1, :])
            shift_labels = Tensor(labels.data[:, 1:])
            loss = self.cross_entropy(shift_logits, shift_labels)
            loss = loss + Tensor(np.array(total_aux / len(self.layers)))

        return logits, loss, total_aux / len(self.layers)

    def cross_entropy(self, logits, targets):
        """Compute cross-entropy loss."""
        B, L, V = logits.data.shape
        logits_2d = logits.reshape(-1, V)
        targets_1d = targets.reshape(-1)
        
        # Stable softmax
        max_vals = logits_2d.data.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits_2d.data - max_vals)
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        
        # Gather probabilities for target tokens
        targets_idx = targets_1d.data.astype(int)
        target_probs = probs[np.arange(len(targets_idx)), targets_idx]
        
        # Mask padding tokens (label = -100)
        valid_mask = (targets_idx != -100).astype(np.float32)
        target_probs = target_probs * valid_mask + (1 - valid_mask) * 1.0
        log_probs = -np.log(target_probs + 1e-8) * valid_mask
        
        loss = log_probs.sum() / max(valid_mask.sum(), 1)
        return Tensor(np.array(loss))

    @staticmethod
    def causal_mask(seq_len):
        """Create causal attention mask."""
        mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1).astype(np.float32)
        return Tensor(mask).reshape(1, 1, seq_len, seq_len)

    @staticmethod
    def generate_ids(model, prompt_ids, max_new=512, temp=0.7, top_k=50, top_p=0.9):
        """Generate tokens autoregressively with KV-cache."""
        model.eval()
        input_t = Tensor(np.array([prompt_ids], dtype=np.int32))
        generated = list(prompt_ids)
        kv_caches = [None] * len(model.layers)

        for step in range(max_new):
            if step == 0:
                x = model.embed(input_t)
            else:
                x = model.embed(Tensor(np.array([[generated[-1]]], dtype=np.int32)))

            for i, layer in enumerate(model.layers):
                if step == 0:
                    h = layer.attn_norm(x)
                    attn_out, kv_caches[i] = layer.attn(h, None, None, 0)
                    x = x + attn_out
                    h = layer.ffn_norm(x)
                    if isinstance(layer.ffn, MoELayer):
                        ffn_out, _ = layer.ffn(h)
                    else:
                        ffn_out = layer.ffn(h)
                    x = x + ffn_out
                else:
                    h = layer.attn_norm(x)
                    attn_out, kv_caches[i] = layer.attn(h, None, kv_caches[i], step)
                    x = x + attn_out
                    h = layer.ffn_norm(x)
                    if isinstance(layer.ffn, MoELayer):
                        ffn_out, _ = layer.ffn(h)
                    else:
                        ffn_out = layer.ffn(h)
                    x = x + ffn_out

            x = model.norm(x)
            logits = model.head(x)

            next_logits = logits.data[0, -1, :] / max(temp, 1e-7)

            # Top-k
            if top_k > 0:
                thresh = np.sort(next_logits)[-top_k]
                next_logits[next_logits < thresh] = -1e9

            # Top-p
            if top_p < 1.0:
                sorted_idx = np.argsort(next_logits)[::-1]
                sorted_logits = next_logits[sorted_idx]
                cum_probs = np.cumsum(np.exp(sorted_logits) / np.exp(sorted_logits).sum())
                remove = cum_probs > top_p
                remove[1:] = remove[:-1].copy()
                remove[0] = False
                sorted_logits[remove] = -1e9
                next_logits[sorted_idx] = sorted_logits

            # Sample
            probs = np.exp(next_logits) / np.exp(next_logits).sum()
            next_token = np.random.choice(len(probs), p=probs)
            generated.append(int(next_token))

            if next_token == 1:  # EOS
                break

        model.train()
        return generated
