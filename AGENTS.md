# AGENTS.md

## Project

Qythera — pure Python + NumPy AI framework. 36 modules, ~17K lines, zero external AI dependencies (only numpy).

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
├── tensor.py          # Autodiff Tensor — foundation for everything
├── nn.py              # Module, Linear, Embedding, RMSNorm, Conv, activations
├── optim.py           # Adam/AdamW/Lion/SAM/etc + LR schedulers
├── positional.py      # RoPE/YaRN/ALiBi
├── model.py           # Transformer (MHA/GQA/MoE/SwiGLU) + generate()
├── tokenizer.py       # BPE/WordPiece/Unigram
├── sampler.py         # TopK/TopP/Beam/Watermark
├── training/          # data pipeline, distillation, quantization
├── peft/              # LoRA/QLoRA/DoRA/VeRA/IA3
├── inference/         # HTTP server, CLI, hardware detection
├── ai/                # agent, reasoning, memory, retrieval, symbolic, planning, logic, world
├── safety/            # PII, jailbreak, output filter, rate limiter
├── eval/              # benchmarks (MMLU, BLEU, ROUGE), interpretability
├── graph/             # GCN, GAT, stats
├── multimodal/        # ViT, CLIP, audio, diffusion
├── systems/           # model merge, distributed, VM, compiler, DSL, knowledge FS
└── web/               # ui.html, sw.js, manifest.json (PWA)
```

## Conventions

- **All imports**: `from qythera.module import Class` (never `from core.` — old flat structure is gone)
- **Backend**: numpy only. No torch, no tensorflow. Tensor class in `tensor.py` has its own autograd.
- **Python 3.9+**: type hints use `dict`/`list` not `Dict`/`List` where possible
- **Default model config**: vocab=32000, embed=256, layers=6, heads=8, kv_heads=4, ffn=768 → ~21M params

## Gotchas

- `PYTHONPATH=src` is required when running from project root without `pip install .`
- `cross_entropy_loss()` expects target shape `(batch,)` not `(batch, seq_len)` — take last token logits first
- `nn.Module.__setattr__` auto-registers `Tensor` and `Module` attributes — don't use `object.__setattr__` for params
- RoPE `_apply_rotary` expects cos/sin with shape `[seq_len, dim//2]` — it duplicates to full dim internally
- `model.generate()` returns a Python list of token IDs, not a Tensor
