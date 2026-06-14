"""Vaelon Transformer Model with Mixture of Experts."""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from vaelon.config import VaelonConfig
from vaelon.layers.attention import VaelonAttention
from vaelon.layers.moe import VaelonMoE
from vaelon.layers.normalization import RMSNorm
from vaelon.layers.ffn import SwiGLU


@dataclass
class ModelOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None


class VaelonDecoderLayer(nn.Module):
    """Single decoder layer with self-attention + MoE FFN."""

    def __init__(self, config: VaelonConfig):
        super().__init__()
        self.self_attn = VaelonAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )
        self.ffn = VaelonMoE(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            bias=config.use_bias,
        )
        self.input_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.post_attention_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[tuple] = None, position_ids: Optional[torch.LongTensor] = None
                ) -> tuple[torch.Tensor, Optional[tuple], dict]:
        residual = x
        x = self.input_norm(x)
        attn_out, new_kv_cache = self.self_attn(x, attention_mask, kv_cache, position_ids)
        x = residual + attn_out

        residual = x
        x = self.post_attention_norm(x)
        ffn_out, aux_info = self.ffn(x)
        x = residual + ffn_out

        return x, new_kv_cache, aux_info


class VaelonModel(nn.Module):
    """Vaelon Transformer with Mixture of Experts.

    A decoder-only transformer with:
    - Grouped Query Attention (GQA)
    - Rotary Position Embeddings (RoPE)
    - Mixture of Experts (MoE) feed-forward layers
    - RMSNorm normalization
    - FlashAttention-3 support
    """

    def __init__(self, config: VaelonConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            VaelonDecoderLayer(config) for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.gradient_checkpointing = False
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        kv_caches: Optional[list] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> ModelOutput:
        hidden_states = self.embed_tokens(input_ids)

        if kv_caches is None:
            kv_caches = [None] * len(self.layers)

        all_aux_loss = 0.0
        new_kv_caches = []

        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                hidden_states, new_cache, aux_info = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, attention_mask, kv_caches[i], position_ids,
                    use_reentrant=False,
                )
            else:
                hidden_states, new_cache, aux_info = layer(
                    hidden_states, attention_mask, kv_caches[i], position_ids,
                )
            new_kv_caches.append(new_cache)
            all_aux_loss = all_aux_loss + aux_info["aux_loss"]

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            aux_weight = 0.01
            loss = loss + aux_weight * all_aux_loss / len(self.layers)

        return ModelOutput(
            loss=loss, logits=logits, hidden_states=hidden_states,
            aux_loss=all_aux_loss / len(self.layers),
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """Generate text autoregressively with KV cache."""
        past_kv_caches = [None] * len(self.layers)
        generated = input_ids.clone()
        cur_len = input_ids.shape[1]

        for _ in range(max_new_tokens):
            if past_kv_caches[0] is not None:
                model_input = input_ids[:, -1:]
                position_ids = torch.tensor([[cur_len - 1]], device=input_ids.device)
            else:
                model_input = input_ids
                position_ids = None

            outputs = self.forward(model_input, position_ids=position_ids,
                                   kv_caches=past_kv_caches)
            past_kv_caches = [None] * len(self.layers)
            next_token_logits = outputs.logits[:, -1, :] / max(temperature, 1e-7)

            if repetition_penalty != 1.0:
                for token_id in generated[0].unique():
                    next_token_logits[:, token_id] /= repetition_penalty

            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1]
                next_token_logits[indices_to_remove] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float("-inf")

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            cur_len += 1

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated
