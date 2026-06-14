"""Tokenizer for Vaelon model."""

import json
import os
from pathlib import Path
from typing import List, Optional

import torch


class VaelonTokenizer:
    """BPE tokenizer wrapper.

    Supports loading from sentencepiece or tiktoken models.
    Falls back to a simple character-level tokenizer if no model is available.
    """

    SPECIAL_TOKENS = {
        "bos": "<|begin_of_text|>",
        "eos": "<|end_of_text|>",
        "pad": "<|pad|>",
        "user": "<|user|>",
        "assistant": "<|assistant|>",
        "system": "<|system|>",
    }

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.vocab_size: int = 0

        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            self._init_default_vocab()

    def _init_default_vocab(self):
        """Initialize a default vocabulary for testing."""
        self.token_to_id = {v: i for i, v in enumerate(self.SPECIAL_TOKENS.values())}
        for i in range(32, 127):
            self.token_to_id[chr(i)] = len(self.token_to_id)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)

    def _load_model(self, path: str):
        vocab_path = Path(path) / "vocab.json"
        if vocab_path.exists():
            with open(vocab_path) as f:
                self.token_to_id = json.load(f)
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}
            self.vocab_size = len(self.token_to_id)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = []
        if add_special_tokens:
            tokens.append(self.token_to_id.get(self.SPECIAL_TOKENS["bos"], 0))
        for char in text:
            tokens.append(self.token_to_id.get(char, self.token_to_id.get(self.SPECIAL_TOKENS["pad"], 0)))
        if add_special_tokens:
            tokens.append(self.token_to_id.get(self.SPECIAL_TOKENS["eos"], 1))
        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        special_ids = set(self.token_to_id.values()) if skip_special_tokens else set()
        for tid in token_ids:
            if tid in special_ids and skip_special_tokens:
                continue
            tokens.append(self.id_to_token.get(tid, ""))
        return "".join(tokens)

    def encode_chat(self, messages: List[dict]) -> List[int]:
        tokens = [self.token_to_id.get(self.SPECIAL_TOKENS["bos"], 0)]
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            role_token = self.SPECIAL_TOKENS.get(role, self.SPECIAL_TOKENS["user"])
            tokens.append(self.token_to_id.get(role_token, 0))
            tokens.extend(self.encode(content, add_special_tokens=False))
        tokens.append(self.token_to_id.get(self.SPECIAL_TOKENS["assistant"], 0))
        return tokens

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "vocab.json"), "w") as f:
            json.dump(self.token_to_id, f)

    def __len__(self) -> int:
        return self.vocab_size
