# Training Guide

## Quick Start
```bash
python -c "from training.trainer import Trainer; Trainer().train('data/training.json')"
```

## Custom Data
Create `data/training.json` with format:
```json
[
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
]
```

## Model Sizes
- Small: 512 hidden, 6 layers, 4 heads (~10M params)
- Medium: 1024 hidden, 12 layers, 8 heads (~100M params)
- Large: 2048 hidden, 24 layers, 16 heads (~1B params)
