# Qythera Model Weights

## Available Models

| Model | Parameters | Experts | File Size | Download |
|-------|-----------|---------|-----------|----------|
| Vaelon-1B | 1B | 2 | ~2GB | Coming soon |
| Vaelon-3B | 3B | 4 | ~6GB | Coming soon |
| Vaelon-7B | 7B | 8 | ~14GB | Coming soon |
| Vaelon-14B | 14B | 8 | ~28GB | Coming soon |
| Vaelon-30B | 30B | 16 | ~60GB | Coming soon |
| Vaelon-70B | 70B | 64 | ~140GB | Coming soon |
| Vaelon-120B | 120B | 128 | ~240GB | Coming soon |

## Training Your Own

```bash
qythera train --config training/configs/7b_lora.yaml
```

## Quantized Versions

AWQ 4-bit and GPTQ 4-bit versions available for efficient inference.
