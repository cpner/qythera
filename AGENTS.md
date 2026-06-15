# AGENTS.md

## Project

Qythera — pure Python + NumPy AI framework. 41 modules, ~25K lines, zero external AI dependencies (only numpy).

## Key Commands

```bash
PYTHONPATH=src python3 -c "from qythera.tensor import Tensor; print('OK')"
PYTHONPATH=src python3 -c "
from qythera.model import Transformer, TransformerConfig
from qythera.tensor import Tensor; import numpy as np
m = Transformer(TransformerConfig())
logits = m(Tensor(np.random.randint(0, 100, (1, 16))))
print(logits.shape)
"
pip install .
PYTHONPATH=src python3 -m qythera.inference.cli serve --port 8080
```

## Structure

```
src/qythera/
├── tensor.py          # Autodiff Tensor — foundation
├── nn.py              # Module, Linear, Embedding, RMSNorm, Conv, activations
├── optim.py           # Adam/AdamW/Lion/SAM/Muon/Sophia + LR schedulers
├── positional.py      # RoPE/YaRN/ALiBi/T5 relative bias
├── model.py           # Transformer + 20+ attention/MoE variants + KV cache
├── tokenizer.py       # BPE/WordPiece/Unigram/Tiktoken
├── sampler.py         # TopK/TopP/Beam/MCTS/DiverseBeam/Contrastive
├── models/            # Mamba, RWKV, xLSTM
├── training/          # trainer, data pipeline, distillation, quantization
├── peft/              # LoRA/QLoRA/DoRA/VeRA/IA3
├── alignment/         # PPO/DPO/SimPO/ORPO/GRPO/KTO/RLAIF/SPIN/ILQL
├── inference/         # HTTP server, CLI, hardware detection
├── ai/                # agent, reasoning, memory, retrieval, symbolic, planning, logic, world
├── safety/            # PII, jailbreak, DP-SGD, adversarial, watermark
├── eval/              # benchmarks, interpretability, profiler
├── graph/             # GCN, GAT, stats
├── multimodal/        # ViT, CLIP, audio, diffusion
├── systems/           # merge, distributed, VM, compiler, DSL, knowledge FS
└── web/               # ui.html, sw.js, manifest.json (PWA)
```

## Conventions

- All imports: `from qythera.module import Class`
- Backend: numpy only. No torch, no tensorflow.
- Python 3.9+
- Default model: vocab=32000, embed=256, layers=6, heads=8, kv_heads=4, ffn=768 → ~21M params

## Gotchas

- `PYTHONPATH=src` required without pip install
- `cross_entropy_loss()` expects target shape `(batch,)` not `(batch, seq_len)`
- nn.Module.__setattr__ auto-registers Tensor and Module attributes
- model.generate() returns Python list of token IDs, not Tensor
