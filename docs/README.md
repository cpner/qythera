# Qythera Documentation

## Overview
Qythera is a production superintelligence platform with the Vaelon model architecture.

## Architecture
- Custom autodiff tensor engine (no torch.nn)
- Vaelon transformer with MoE, GQA, RoPE
- BPE tokenizer (from scratch)
- Training pipeline (pretrain, SFT, DPO)
- Inference server (OpenAI-compatible API)
- Memory system (vector index + episodic)
- Safety filters (toxicity, jailbreak, PII)
- Web UI (Next.js, glassmorphism, mobile-responsive)
- CLI interface

## Quick Start
```bash
pip install -e .
python -m inference.server
cd web && npm install && npm run dev
```
