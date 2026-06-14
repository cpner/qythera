# Training Guide

## Pre-training

```bash
torchrun --nproc_per_node=4 training/pretrain/train_pretrain.py \
  --config training/configs/7b_lora.yaml
```

## Fine-tuning (SFT)

```bash
python training/finetune/sft_trainer.py
```

## RLHF (DPO)

```bash
python training/rlhf/dpo_trainer.py
```

## Model Sizes

| Model | Parameters | Experts | Hidden | Layers | Heads |
|-------|-----------|---------|--------|--------|-------|
| 7B | 7B | 8 | 4096 | 32 | 32 |
| 13B | 13B | 8 | 5120 | 40 | 40 |
| 70B | 70B | 64 | 8192 | 80 | 64 |

## Hardware Requirements

- **7B**: 1x A100 40GB or 2x RTX 4090
- **13B**: 2x A100 80GB
- **70B**: 8x A100 80GB
