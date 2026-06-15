# AGENTS.md

## Project

Qythera — pure Python + NumPy AI framework. 45 modules, 29,580 lines, zero external AI dependencies (only numpy, optional). Dual backend: pure Python when NumPy unavailable, NumPy when available.

## Key Commands

```bash
# Run anything (src layout requires PYTHONPATH)
PYTHONPATH=src python3 -c "from qythera.tensor import Tensor; print('OK')"

# Test the full model
PYTHONPATH=src python3 -c "
from qythera.model import Transformer, TransformerConfig
from qythera.tensor import Tensor; import numpy as np
m = Transformer(TransformerConfig())
logits = m(Tensor(np.random.randint(0, 100, (1, 16))))
print(logits.shape)
"

# Install as package
pip install .

# Start server (needs PYTHONPATH or pip install)
PYTHONPATH=src python3 -m qythera.inference.cli serve --port 8080
```

## Structure

```
src/qythera/
├── backend.py          # Dual backend (pure Python + NumPy)
├── tensor.py           # Autodiff engine (2,327 lines)
├── nn.py               # Module, Linear, Embedding, RMSNorm, Conv, activations
├── optim.py            # 26 optimizers + 10 LR schedulers
├── positional.py       # RoPE/YaRN/ALiBi
├── model.py            # Transformer + 30+ attention/MoE variants + KV cache
├── tokenizer.py        # BPE/WordPiece/Unigram/Tiktoken
├── sampler.py          # TopK/TopP/Beam/MCTS/DiverseBeam/Contrastive
├── models/             # Mamba, RWKV, xLSTM
├── training/           # trainer, data pipeline, distillation, quantization
├── peft/               # LoRA/QLoRA/DoRA/VeRA/IA3
├── alignment/          # PPO/DPO/SimPO/ORPO/GRPO/KTO/RLAIF/SPIN/ILQL/SelfPlay
├── inference/          # HTTP server, CLI, hardware detection
├── ai/                 # 12 modules: reasoning, memory, retrieval, symbolic, planning, logic, world, scientific, probabilistic, theoretical, agent, knowledge
├── safety/             # PII, jailbreak, DP-SGD, adversarial, watermark
├── eval/               # benchmarks, interpretability, profiler
├── graph/              # GCN, GAT, stats
├── multimodal/         # ViT, CLIP, audio
├── systems/            # merge, distributed, VM, compiler, DSL, knowledge_fs
└── web/                # ui.html, sw.js, manifest.json (PWA)
```

## Conventions

- All imports: `from qythera.module import Class`
- Backend: pure Python by default, NumPy when available
- Python 3.9+
- Default model: vocab=32000, embed=256, layers=6, heads=8, kv_heads=4, ffn=768 → ~21M params

## Gotchas

- `PYTHONPATH=src` required without pip install
- `cross_entropy_loss()` expects target shape `(batch,)` not `(batch, seq_len)`
- nn.Module.__setattr__ auto-registers Tensor and Module attributes
- model.generate() returns Python list of token IDs, not Tensor
- Tensor indexing requires int64 — arange returns float32
- Embedding backward requires _ctx with EmbeddingBackward
- Dropout must preserve autograd graph — use Tensor ops, not numpy
