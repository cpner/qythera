"""Download and cache training datasets from HuggingFace."""

import os
import hashlib
import json
from pathlib import Path
from typing import Optional

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


DATASET_REGISTRY = {
    "openhermes": {
        "hf_id": "teknium/OpenHermes-2.5",
        "split": "train",
        "fields": {"messages": "messages"},
    },
    "ultrachat": {
        "hf_id": "stingning/ultrachat",
        "split": "train",
        "fields": {"messages": "prompt", "response": "response"},
    },
    "wildchat": {
        "hf_id": "lldwd/wildchat",
        "split": "train",
        "fields": {"messages": "conversation"},
    },
}


def download_dataset(name: str, cache_dir: str = "./data/cache",
                     max_samples: Optional[int] = None,
                     force_download: bool = False) -> list:
    """Download a dataset by name from the registry.

    Args:
        name: Dataset name (openhermes, ultrachat, wildchat)
        cache_dir: Directory to cache downloaded data
        max_samples: Maximum number of samples to load
        force_download: Force re-download even if cached

    Returns:
        List of samples
    """
    if load_dataset is None:
        raise ImportError("Install datasets: pip install datasets")

    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")

    config = DATASET_REGISTRY[name]
    cache_path = Path(cache_dir) / name
    cache_file = cache_path / "samples.json"

    if cache_file.exists() and not force_download:
        print(f"Loading cached {name} from {cache_file}")
        with open(cache_file) as f:
            data = json.load(f)
        if max_samples:
            data = data[:max_samples]
        return data

    print(f"Downloading {config['hf_id']}...")
    dataset = load_dataset(config["hf_id"], split=config["split"], cache_dir=str(cache_path))

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    samples = [dict(row) for row in dataset]

    cache_path.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(samples, f)

    print(f"Downloaded {len(samples)} samples from {name}")
    return samples


def download_all(cache_dir: str = "./data/cache", max_samples_per: Optional[int] = None):
    """Download all registered datasets."""
    results = {}
    for name in DATASET_REGISTRY:
        try:
            results[name] = download_dataset(name, cache_dir, max_samples_per)
        except Exception as e:
            print(f"Failed to download {name}: {e}")
            results[name] = []
    return results
