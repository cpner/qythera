# Qythera Architecture

## Overview

Qythera is built on the **Vaelon** model architecture - a decoder-only transformer with Mixture of Experts (MoE).

## Model Architecture (Vaelon)

```
Input Tokens
    |
[Token Embeddings] + [RoPE Position Embeddings]
    |
[VaelonDecoderLayer] x N layers
    |-- RMSNorm
    |-- Grouped Query Attention (GQA) with FlashAttention
    |-- Residual Connection
    |-- RMSNorm
    |-- Mixture of Experts (MoE) FFN
    |-- Residual Connection
    |
[RMSNorm]
    |
[LM Head] -> Logits
```

### Key Components

1. **Grouped Query Attention (GQA)**: Reduces KV-cache by sharing key-value heads across query groups
2. **Rotary Position Embeddings (RoPE)**: Position encoding without learned parameters
3. **Mixture of Experts (MoE)**: Routes tokens to top-k experts for efficient scaling
4. **RMSNorm**: More stable normalization than LayerNorm
5. **SwiGLU**: Gated activation function for FFN layers

## System Components

- **Training**: DeepSpeed ZeRO-3, LoRA/QLoRA, DPO/PPO
- **Inference**: vLLM with AWQ/GPTQ quantization
- **Memory**: FAISS vector store + episodic memory
- **Agent**: ReAct reasoning loop with tool use
- **Safety**: Toxicity, jailbreak, and PII filtering
- **Web UI**: Next.js 14 with glassmorphism design
- **CLI**: Click-based with Rich terminal UI
