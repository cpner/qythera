# Qythera

Production Superintelligence Platform. Works everywhere.

## Quick Start
```bash
git clone https://github.com/cpner/qythera.git
cd qythera
pip install numpy
python -m core.server --port 8080
```

Open http://localhost:8080 in any browser.

## Features
- Custom autodiff tensor engine
- Transformer with MoE, GQA, RoPE
- BPE tokenizer
- 50+ knowledge topics, 7 code templates, math engine
- Safety filters
- Works on Linux, macOS, Windows, Android, Docker
- No external APIs needed

## Commands
```bash
python -m core.server --port 8080    # Start server
python -c "from core.model import Model; print('OK')"  # Test model
```

## License
MIT
