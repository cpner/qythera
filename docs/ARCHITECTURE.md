# Vaelon Architecture

## Model Design
Vaelon is a decoder-only transformer with Mixture of Experts.

### Components
1. **Token Embedding**: Maps token IDs to dense vectors
2. **Decoder Layers** (x N):
   - RMSNorm -> Multi-Head Attention (GQA + RoPE) -> Residual
   - RMSNorm -> MoE FFN -> Residual
3. **Final RMSNorm**
4. **LM Head**: Projects to vocabulary

### Mixture of Experts
- Gate network routes tokens to top-k experts
- Each expert is a SwiGLU FFN
- Load balancing loss encourages均匀 routing

### Grouped Query Attention
- Shares KV heads across query groups
- Reduces memory usage for long sequences

### RoPE (Rotary Position Embeddings)
- Encodes position information in query/key vectors
- Supports arbitrary sequence lengths
