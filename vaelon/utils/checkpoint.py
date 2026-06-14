"""Model checkpoint save/load utilities."""

import os
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def save_checkpoint(model: nn.Module, optimizer, step: int, path: str,
                    config: Optional[dict] = None, extra: Optional[dict] = None):
    """Save model checkpoint with sharding support."""
    os.makedirs(path, exist_ok=True)
    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
    }
    if extra:
        state.update(extra)
    torch.save(state, os.path.join(path, "checkpoint.pt"))
    if config:
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)


def load_checkpoint(path: str, model: nn.Module, optimizer=None, device: str = "cpu"):
    """Load model checkpoint."""
    ckpt_path = os.path.join(path, "checkpoint.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    if optimizer and state.get("optimizer_state_dict"):
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state.get("step", 0), state


def save_sharded(model: nn.Module, path: str, max_shard_size: int = 5_000_000_000):
    """Save model in shards for large models."""
    os.makedirs(path, exist_ok=True)
    state_dict = model.state_dict()
    shard_idx = 0
    current_shard = {}
    current_size = 0
    for key, tensor in state_dict.items():
        tensor_size = tensor.nelement() * tensor.element_size()
        if current_size + tensor_size > max_shard_size and current_shard:
            torch.save(current_shard, os.path.join(path, f"shard-{shard_idx:05d}.pt"))
            shard_idx += 1
            current_shard = {}
            current_size = 0
        current_shard[key] = tensor
        current_size += tensor_size
    if current_shard:
        torch.save(current_shard, os.path.join(path, f"shard-{shard_idx:05d}.pt"))
    index = {"metadata": {"total_size": sum(t.nelement() * t.element_size() for t in state_dict.values())},
             "weight_map": {k: f"shard-{i:05d}.pt" for i, (k, _) in enumerate(state_dict.items())}}
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
