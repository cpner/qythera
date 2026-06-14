# Architecture

## Vaelon Transformer

- Token Embedding + Positional Encoding
- N Decoder Layers: RMSNorm -> Multi-Head Attention (GQA, RoPE) -> RMSNorm -> SwiGLU FFN
- Final RMSNorm -> LM Head

## Key Features
- Grouped Query Attention (GQA)
- Rotary Position Embeddings (RoPE)
- RMSNorm normalization
- SwiGLU activation
- KV Cache for fast inference
- MoE (optional)

## Model Sizes
| Name | d_model | Layers | Heads | Params |
|------|---------|--------|-------|--------|
| Tiny | 64 | 2 | 4 | ~50K |
| Small | 128 | 4 | 8 | ~500K |
| Medium | 256 | 6 | 8 | ~3M |
| Large | 512 | 8 | 16 | ~20M |
| XLarge | 1024 | 12 | 32 | ~150M |
