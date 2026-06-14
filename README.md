# Qythera

Production Superintelligence Platform. Pure Python + NumPy. No external AI APIs.

## Quick Start
```bash
git clone https://github.com/cpner/qythera.git
cd qythera
pip install numpy
python3 -m core.cli serve --port 8080
```

Open http://localhost:8080 in any browser.

## Architecture
```
tokens -> embeddings -> N transformer layers -> logits -> sampling
Each layer: RMSNorm -> MHA+RoPE -> RMSNorm -> SwiGLU -> residual
```

## Modules (37 files, 16,697 lines)

### Core Engine
- tensor.py - Custom autodiff tensor engine: 50+ ops, broadcasting, einsum, SVD, QR, Cholesky, LU, eig
- nn.py - Neural network modules: Linear, Embedding, RMSNorm, LayerNorm, BatchNorm, Conv1D/2D/3D, 15 activations
- optim.py - 18 optimizers: Adam, AdamW, Lion, Muon, SOAP, Sophia, SAM + 10 LR schedulers
- positional.py - RoPE, YaRN, NTK-aware, LongRoPE, ALiBi, Sinusoidal, T5 bias, FIRE

### Model
- model.py - Full transformer: MHA/GQA/MQA, SwiGLU FFN, MoE, KV cache, generate()
- tokenizer.py - BPE, WordPiece, Unigram, Tiktoken-style, chat template, FIM
- sampler.py - Greedy, TopK, TopP, MinP, Beam Search, Self-Consistency, Watermarking

### Training & Optimization
- data.py - MMapDataset, StreamingDataset, DataLoader, MinHash dedup, SimHash
- quantize.py - INT8/INT4/INT2, GPTQ, AWQ, SmoothQuant, LLM.int8
- peft.py - LoRA, QLoRA, AdaLoRA, DoRA, VeRA, PrefixTuning, IA3, Adapter, PromptTuning
- distill.py - Knowledge distillation, Medusa heads, MultiToken prediction, Speculative decoding
- merge.py - Task Arithmetic, TIES, DARE, SLERP, Model Soup

### AI Capabilities
- retrieval.py - BM25, Dense retrieval, Hybrid, ColBERT, InvertedIndex
- agent.py - Tool calling, ReACT loop, Reflexion, Multi-Agent Debate
- reasoning.py - Chain-of-Thought, Self-Consistency, Tree-of-Thought, Constitutional AI
- memory.py - Episodic, Semantic, Working, Procedural, Long-term, Hopfield
- symbolic.py - Knowledge Graph, SAT solver, First-Order Logic, Symbolic Regression
- planning.py - A*, MCTS, IDA*, STRIPS planning
- logic.py - Propositional, First-Order, Modal, Temporal, Fuzzy logic
- world.py - Physics simulation, Causal DAG, BDI agents
- graph_nn.py - GCN, GraphSAGE, GAT, TransE
- stats.py - Entropy, KL/JS divergence, MI, Bayesian, EM, hypothesis tests

### Multimodal & Safety
- multimodal.py - ViT, CLIP, Audio STFT+mel, Simple Diffusion
- safety.py - Jailbreak detection, PII, Output filtering, Rate limiting, Watermark
- evaluate.py - MMLU, HumanEval, GSM8K, BLEU, ROUGE, ECE, Arena ELO
- interpret.py - Attention viz, Integrated Gradients, Logit Lens, Causal Tracing

### Infrastructure
- server.py - HTTP server, OpenAI API, SSE streaming, batch scheduling
- cli.py - CLI: train, infer, tokenize, quantize, serve, evaluate
- hardware.py - CPU/RAM/SIMD detection, auto-config
- distributed.py - Data/Tensor Parallel, ZeRO 1/2/3, gradient compression
- vm.py - 15-opcode Tensor VM, memory manager, profiler
- compiler.py - Graph IR, constant folding, DCE, operator fusion, JIT
- language.py - DSL lexer/parser/codegen for model definitions
- knowledge_fs.py - Virtual filesystem, HNSW index, URI scheme
- profiler.py - cProfile, memory profiler, AutoML, model analyzer

## Default Model
~21M params: vocab=32000, embed=256, layers=6, heads=8, kv_heads=4, ffn=768, ctx=2048

## API (OpenAI Compatible)
POST /v1/chat/completions | POST /v1/completions | POST /v1/embeddings | GET /health

## License
MIT
