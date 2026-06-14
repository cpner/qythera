"""Token and positional embeddings."""

import numpy as np


class TokenEmbedding:
    def __init__(self, vocab_size, d_model):
        self.weight = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.02
    
    def forward(self, ids):
        return self.weight[ids]


class PositionalEncoding:
    def __init__(self, d_model, max_seq=2048):
        self.weight = np.random.randn(max_seq, d_model).astype(np.float32) * 0.01
    
    def forward(self, seq_len):
        return self.weight[:seq_len]
