import os
import json
from typing import List, Dict, Optional

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


def get_tokenizer(model_name_or_path=None):
    if AutoTokenizer and model_name_or_path:
        return AutoTokenizer.from_pretrained(model_name_or_path)
    return None


def format_as_text(messages):
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"[{role.upper()}]\n{content}")
    return "\n\n".join(parts)


def tokenize_dataset(samples, tokenizer, max_length=2048):
    tokenized = []
    for sample in samples:
        messages = sample.get("messages", [])
        if not messages:
            continue
        text = format_as_text(messages)
        ids = tokenizer.encode(text, truncation=True, max_length=max_length)
        if len(ids) > 10:
            tokenized.append({"input_ids": ids, "labels": ids.copy()})
    return tokenized


def pack_sequences(tokenized_samples, max_length=2048):
    packed = []
    current = []
    current_len = 0
    for sample in tokenized_samples:
        ids = sample["input_ids"]
        if current_len + len(ids) <= max_length:
            current.extend(ids)
            current_len += len(ids)
        else:
            if current:
                packed.append({"input_ids": current, "labels": current.copy()})
            current = ids[:max_length]
            current_len = max_length
    if current:
        packed.append({"input_ids": current, "labels": current.copy()})
    return packed
