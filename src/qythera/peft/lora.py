"""Parameter-Efficient Fine-Tuning methods. Pure Python + NumPy."""
import numpy as np
import math
from collections import OrderedDict

from qythera.tensor import Tensor, no_grad, randn, zeros, ones, eye
from qythera.nn import Module, Linear


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class LoRA(Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        sigma = math.sqrt(2.0 / (in_features + rank))
        self.base_weight = Tensor(np.random.randn(out_features, in_features).astype(np.float32) * math.sqrt(2.0 / in_features))
        self.A = Tensor((np.random.randn(rank, in_features) * sigma).astype(np.float32))
        self.B = Tensor(np.zeros((out_features, rank), dtype=np.float32))
        self._merged = False

    def forward(self, x):
        if self._merged:
            return x.matmul(self.base_weight.T)
        delta = self.B.matmul(self.A) * self.scaling
        return x.matmul((self.base_weight + delta).T)

    def merge(self):
        if not self._merged:
            delta = self.B.matmul(self.A) * self.scaling
            self.base_weight = self.base_weight + delta
            self._merged = True

    def unmerge(self):
        if self._merged:
            delta = self.B.matmul(self.A) * self.scaling
            self.base_weight = self.base_weight - delta
            self._merged = False


# ---------------------------------------------------------------------------
# QLoRA — INT4 quantized base + FP32 LoRA adapters
# ---------------------------------------------------------------------------

def _quantize_int4(arr):
    flat = arr.flatten()
    scale = np.max(np.abs(flat)) / 7.0 + 1e-8
    q = np.clip(np.round(flat / scale), -8, 7).astype(np.int8)
    n = len(q)
    packed = bytearray((n + 1) // 2)
    for i in range(n):
        v = int(q[i]) & 0x0F
        if i % 2 == 0:
            packed[i // 2] = v
        else:
            packed[i // 2] |= v << 4
    return bytes(packed), scale, arr.shape


def _dequantize_int4(packed, scale, shape):
    n = 1
    for s in shape:
        n *= s
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        b = packed[i // 2]
        val = (b & 0x0F) if (i % 2 == 0) else ((b >> 4) & 0x0F)
        if val > 7:
            val -= 16
        out[i] = val * scale
    return out.reshape(shape)


class QLoRA(Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        weight_fp = np.random.randn(out_features, in_features).astype(np.float32) * math.sqrt(2.0 / in_features)
        packed, scale, shape = _quantize_int4(weight_fp)
        self.register_buffer('base_packed', Tensor(np.frombuffer(packed, dtype=np.uint8).astype(np.float32)))
        self.register_buffer('base_scale', Tensor(np.array([scale], dtype=np.float32)))
        self._base_shape = shape
        sigma = math.sqrt(2.0 / (in_features + rank))
        self.A = Tensor((np.random.randn(rank, in_features) * sigma).astype(np.float32))
        self.B = Tensor(np.zeros((out_features, rank), dtype=np.float32))
        self._merged = False

    def _dequantized_weight(self):
        packed = self.base_packed.data.astype(np.uint8).tobytes()
        return _dequantize_int4(packed, float(self.base_scale.data[0]), self._base_shape)

    def forward(self, x):
        if self._merged:
            w = Tensor(self._dequantized_weight(), requires_grad=False)
            return x.matmul(w.T)
        base = Tensor(self._dequantized_weight(), requires_grad=False)
        delta = self.B.matmul(self.A) * self.scaling
        return x.matmul((base + delta).T)

    def merge(self):
        if not self._merged:
            delta = self.B.matmul(self.A) * self.scaling
            base_fp = self._dequantized_weight() + delta.data
            packed, scale, shape = _quantize_int4(base_fp)
            self.base_packed.data = np.frombuffer(packed, dtype=np.uint8).astype(np.float32)
            self.base_scale.data = np.array([scale], dtype=np.float32)
            self._base_shape = shape
            self._merged = True

    def unmerge(self):
        if self._merged:
            base_fp = self._dequantized_weight()
            delta = self.B.matmul(self.A) * self.scaling
            orig_fp = base_fp - delta.data
            packed, scale, shape = _quantize_int4(orig_fp)
            self.base_packed.data = np.frombuffer(packed, dtype=np.uint8).astype(np.float32)
            self.base_scale.data = np.array([scale], dtype=np.float32)
            self._base_shape = shape
            self._merged = False


# ---------------------------------------------------------------------------
# AdaLoRA — importance-based SVD pruning of LoRA adapters
# ---------------------------------------------------------------------------

class AdaLoRA(Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0, target_rank=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.target_rank = target_rank or rank
        self.base_weight = Tensor(np.random.randn(out_features, in_features).astype(np.float32) * math.sqrt(2.0 / in_features))
        self.P = Tensor(np.random.randn(out_features, rank).astype(np.float32) * math.sqrt(2.0 / out_features))
        self.Lambda = Tensor(np.ones(rank, dtype=np.float32))
        self.Q = Tensor(np.random.randn(rank, in_features).astype(np.float32) * math.sqrt(2.0 / in_features))
        self.importance = Tensor(np.zeros(rank, dtype=np.float32))
        self._merged = False

    def forward(self, x):
        if self._merged:
            return x.matmul(self.base_weight.T)
        delta = (self.P * self.Lambda.unsqueeze(0)).matmul(self.Q) * self.scaling
        return x.matmul((self.base_weight + delta).T)

    def update_importance(self, grad_P, grad_Lambda, grad_Q):
        imp = (self.Lambda.data ** 2) * (
            (self.P.data ** 2).sum(axis=0) * (self.Q.data ** 2).sum(axis=1)
        )
        self.importance.data = imp

    def prune(self):
        sorted_idx = np.argsort(self.importance.data)[::-1]
        keep = sorted_idx[:self.target_rank]
        self.P.data = self.P.data[:, keep]
        self.Lambda.data = self.Lambda.data[keep]
        self.Q.data = self.Q.data[keep, :]
        self.importance.data = self.importance.data[keep]
        self.rank = self.target_rank

    def merge(self):
        if not self._merged:
            delta = (self.P * self.Lambda.unsqueeze(0)).matmul(self.Q) * self.scaling
            self.base_weight = self.base_weight + delta
            self._merged = True

    def unmerge(self):
        if self._merged:
            delta = (self.P * self.Lambda.unsqueeze(0)).matmul(self.Q) * self.scaling
            self.base_weight = self.base_weight - delta
            self._merged = False


# ---------------------------------------------------------------------------
# DoRA — Weight-Decomposed Low-Rank Adaptation
# ---------------------------------------------------------------------------

class DoRA(Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.base_weight = Tensor(np.random.randn(out_features, in_features).astype(np.float32) * math.sqrt(2.0 / in_features))
        norm = np.sqrt((self.base_weight.data ** 2).sum(axis=1, keepdims=True)) + 1e-8
        self.magnitude = Tensor(norm.squeeze().copy())
        V = self.base_weight.data / norm
        self.V = Tensor(V.copy())
        sigma = math.sqrt(2.0 / (in_features + rank))
        self.A = Tensor((np.random.randn(rank, in_features) * sigma).astype(np.float32))
        self.B = Tensor(np.zeros((out_features, rank), dtype=np.float32))

    def forward(self, x):
        delta = self.B.matmul(self.A) * self.scaling
        V_adapted = self.V + delta
        norm = V_adapted.norm(ord=2, axis=1, keepdims=True).data + 1e-8
        W = self.magnitude.unsqueeze(1) * (V_adapted / Tensor(norm))
        return x.matmul(W.T)


# ---------------------------------------------------------------------------
# VeRA — Vector-based Random Matrix Adaptation
# ---------------------------------------------------------------------------

class VeRA(Module):
    _shared_A = None
    _shared_B = None
    _shared_in = None
    _shared_out = None

    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.base_weight = Tensor(np.random.randn(out_features, in_features).astype(np.float32) * math.sqrt(2.0 / in_features))

        if VeRA._shared_A is None or VeRA._shared_in != in_features or VeRA._shared_out != out_features:
            VeRA._shared_A = Tensor((np.random.randn(rank, in_features) * math.sqrt(2.0 / (in_features + rank))).astype(np.float32))
            VeRA._shared_B = Tensor((np.random.randn(out_features, rank) * math.sqrt(2.0 / (out_features + rank))).astype(np.float32))
            VeRA._shared_A.requires_grad_(False)
            VeRA._shared_B.requires_grad_(False)
            VeRA._shared_in = in_features
            VeRA._shared_out = out_features

        self.d = Tensor(np.ones(in_features, dtype=np.float32))
        self.b = Tensor(np.ones(out_features, dtype=np.float32))
        self._merged = False

    def forward(self, x):
        if self._merged:
            return x.matmul(self.base_weight.T)
        A_eff = VeRA._shared_A * self.d.unsqueeze(0)
        B_eff = VeRA._shared_B * self.b.unsqueeze(1)
        delta = B_eff.matmul(A_eff) * self.scaling
        return x.matmul((self.base_weight + delta).T)

    def merge(self):
        if not self._merged:
            A_eff = VeRA._shared_A * self.d.unsqueeze(0)
            B_eff = VeRA._shared_B * self.b.unsqueeze(1)
            delta = B_eff.matmul(A_eff) * self.scaling
            self.base_weight = self.base_weight + delta
            self._merged = True

    def unmerge(self):
        if self._merged:
            A_eff = VeRA._shared_A * self.d.unsqueeze(0)
            B_eff = VeRA._shared_B * self.b.unsqueeze(1)
            delta = B_eff.matmul(A_eff) * self.scaling
            self.base_weight = self.base_weight - delta
            self._merged = False


# ---------------------------------------------------------------------------
# PrefixTuning — learnable key/value prefixes
# ---------------------------------------------------------------------------

class PrefixTuning(Module):
    def __init__(self, num_heads, head_dim, num_prefix=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_prefix = num_prefix
        total_dim = num_heads * head_dim
        self.P_k = Tensor(np.random.randn(num_prefix, total_dim).astype(np.float32) * 0.02)
        self.P_v = Tensor(np.random.randn(num_prefix, total_dim).astype(np.float32) * 0.02)

    def forward(self, K, V):
        B = K.shape[0]
        prefix_k = self.P_k.unsqueeze(0).reshape(1, self.num_prefix, -1).data.repeat(B, axis=0)
        prefix_v = self.P_v.unsqueeze(0).reshape(1, self.num_prefix, -1).data.repeat(B, axis=0)
        K_cat = Tensor(np.concatenate([prefix_k, K.data], axis=1), requires_grad=K.requires_grad or self.P_k.requires_grad)
        V_cat = Tensor(np.concatenate([prefix_v, V.data], axis=1), requires_grad=V.requires_grad or self.P_v.requires_grad)
        return K_cat, V_cat


# ---------------------------------------------------------------------------
# AdapterLayers — bottleneck adapter with residual
# ---------------------------------------------------------------------------

class AdapterLayers(Module):
    def __init__(self, hidden_dim, adapter_dim=64, activation='relu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.adapter_dim = adapter_dim
        self.down_proj = Linear(hidden_dim, adapter_dim, bias=True)
        self.up_proj = Linear(adapter_dim, hidden_dim, bias=True)
        self.activation_name = activation

    def _activate(self, x):
        if self.activation_name == 'relu':
            return Tensor(np.maximum(x.data, 0), requires_grad=x.requires_grad)
        elif self.activation_name == 'gelu':
            return x * Tensor(0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x.data + 0.044715 * x.data ** 3))), requires_grad=x.requires_grad)
        elif self.activation_name == 'silu':
            return x * Tensor(1.0 / (1.0 + np.exp(-x.data)), requires_grad=x.requires_grad)
        return x

    def forward(self, x):
        residual = x
        h = self.down_proj(x)
        h = self._activate(h)
        h = self.up_proj(h)
        return residual + h


# ---------------------------------------------------------------------------
# IA3 — Infused Adapter by Inhibiting and Amplifying Inner Activations
# ---------------------------------------------------------------------------

class IA3(Module):
    def __init__(self, key_dim, value_dim, ffn_dim):
        super().__init__()
        self.l_k = Tensor(np.ones(key_dim, dtype=np.float32))
        self.l_v = Tensor(np.ones(value_dim, dtype=np.float32))
        self.l_ff = Tensor(np.ones(ffn_dim, dtype=np.float32))

    def scale_keys(self, K):
        return K * self.l_k

    def scale_values(self, V):
        return V * self.l_v

    def scale_ffn(self, x):
        return x * self.l_ff


# ---------------------------------------------------------------------------
# PromptTuning — learnable soft prompt tokens
# ---------------------------------------------------------------------------

class PromptTuning(Module):
    def __init__(self, vocab_size, embedding_dim, num_prompts=8):
        super().__init__()
        self.num_prompts = num_prompts
        self.embedding_dim = embedding_dim
        self.soft_tokens = Tensor(np.random.randn(num_prompts, embedding_dim).astype(np.float32) * 0.02)

    def forward(self, input_embeddings):
        B = input_embeddings.shape[0]
        prompt = self.soft_tokens.data.reshape(1, self.num_prompts, -1).repeat(B, axis=0)
        return Tensor(np.concatenate([prompt, input_embeddings.data], axis=1),
                      requires_grad=input_embeddings.requires_grad or self.soft_tokens.requires_grad)


# ---------------------------------------------------------------------------
# LoRAManager — inject, manage, merge LoRA across a model
# ---------------------------------------------------------------------------

class LoRAManager:
    def __init__(self, model, target_modules=None, rank=4, alpha=1.0):
        self.model = model
        self.target_modules = target_modules or ['q_proj', 'v_proj', 'k_proj', 'out_proj', 'fc1', 'fc2']
        self.rank = rank
        self.alpha = alpha
        self.lora_modules = {}
        self._original_weights = {}
        self._inject()

    def _inject(self):
        for name, module in self.model.named_modules():
            if isinstance(module, Linear) and any(t in name for t in self.target_modules):
                lora = LoRA(module.in_features, module.out_features, self.rank, self.alpha)
                lora.base_weight = module.weight.clone()
                self._original_weights[name] = module.weight.clone()
                self.lora_modules[name] = lora
                module._lora = lora

    def merge_all(self):
        for name, lora in self.lora_modules.items():
            lora.merge()
            self._set_weight(name, lora.base_weight)

    def unmerge_all(self):
        for name, lora in self.lora_modules.items():
            lora.unmerge()
            self._set_weight(name, lora.base_weight)

    def _set_weight(self, name, weight):
        parts = name.split('.')
        mod = self.model
        for p in parts:
            mod = mod._modules[p]
        mod.weight = weight

    def trainable_params(self):
        total = 0
        trainable = 0
        for p in self.model.parameters():
            total += p.data.size
        for name, lora in self.lora_modules.items():
            trainable += lora.A.data.size + lora.B.data.size
        return trainable, total

    def trainable_ratio(self):
        trainable, total = self.trainable_params()
        return trainable / total if total > 0 else 0.0

    def state_dict(self):
        state = {}
        for name, lora in self.lora_modules.items():
            state[f"{name}.A"] = lora.A.data.copy()
            state[f"{name}.B"] = lora.B.data.copy()
            state[f"{name}.base_weight"] = lora.base_weight.data.copy()
        return state

    def load_state_dict(self, state_dict):
        for name, lora in self.lora_modules.items():
            a_key = f"{name}.A"
            b_key = f"{name}.B"
            bw_key = f"{name}.base_weight"
            if a_key in state_dict:
                lora.A.data = state_dict[a_key]
            if b_key in state_dict:
                lora.B.data = state_dict[b_key]
            if bw_key in state_dict:
                lora.base_weight.data = state_dict[bw_key]
