"""Dataset and DataLoader for training."""

import json
import os
import numpy as np
from typing import List, Dict, Optional


class Dataset:
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @classmethod
    def from_text(cls, text, tokenizer, seq_len=256):
        ids = tokenizer.encode(text, add_special=False)
        samples = []
        for i in range(0, len(ids) - seq_len - 1, seq_len // 2):
            samples.append(ids[i:i + seq_len + 1])
        return cls(samples)
    
    @classmethod
    def from_file(cls, path, tokenizer, seq_len=256):
        with open(path) as f:
            text = f.read()
        return cls.from_text(text, tokenizer, seq_len)


class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = [self.dataset[j] for j in batch_indices]
            max_len = max(len(s) for s in batch)
            padded = [s + [0] * (max_len - len(s)) for s in batch]
            input_ids = np.array([s[:-1] for s in padded], dtype=np.int64)
            targets = np.array([s[1:] for s in padded], dtype=np.int64)
            yield input_ids, targets
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
