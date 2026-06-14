"""Byte Pair Encoding tokenizer from scratch."""

import json
import os
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional


class BPETokenizer:
    """BPE tokenizer trained from scratch on text data.
    
    Algorithm:
    1. Start with character-level vocabulary
    2. Iteratively merge most frequent adjacent pairs
    3. Build vocabulary up to target size
    """
    
    SPECIAL = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "<sep>": 4, "<cls>": 5}
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []
        self.char_to_id: Dict[str, int] = dict(self.SPECIAL)
        self.id_to_char: Dict[int, str] = {v: k for k, v in self.SPECIAL.items()}
    
    def _get_pair_stats(self, ids_list: List[List[int]]) -> Counter:
        """Count frequency of adjacent pairs across all sequences."""
        stats = Counter()
        for ids in ids_list:
            for i in range(len(ids) - 1):
                stats[(ids[i], ids[i+1])] += 1
        return stats
    
    def _merge_pair(self, ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """Merge all occurrences of a pair in a sequence."""
        result = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                result.append(new_id)
                i += 2
            else:
                result.append(ids[i])
                i += 1
        return result
    
    def train(self, texts: List[str], vocab_size: Optional[int] = None, verbose: bool = False):
        """Train BPE tokenizer on a list of texts."""
        if vocab_size:
            self.vocab_size = vocab_size
        
        # Initialize with character vocabulary
        chars = set()
        for text in texts:
            chars.update(text)
        
        for c in sorted(chars):
            if c not in self.char_to_id:
                self.char_to_id[c] = len(self.char_to_id)
        
        # Convert texts to character IDs
        all_ids = []
        for text in texts:
            ids = [self.char_to_id.get(c, self.char_to_id["<unk>"]) for c in text]
            all_ids.append(ids)
        
        # Iteratively merge most frequent pairs
        num_merges = self.vocab_size - len(self.char_to_id)
        for i in range(num_merges):
            stats = self._get_pair_stats(all_ids)
            if not stats:
                break
            
            best_pair = max(stats, key=stats.get)
            if stats[best_pair] < 2:
                break
            
            new_id = len(self.char_to_id)
            self.merges.append(best_pair)
            self.char_to_id[f"<merge_{i}>"] = new_id
            self.id_to_char[new_id] = f"<merge_{i}>"
            
            all_ids = [self._merge_pair(ids, best_pair, new_id) for ids in all_ids]
            
            if verbose and (i + 1) % 100 == 0:
                print(f"  Merge {i+1}/{num_merges}: {best_pair} -> {new_id} (freq={stats[best_pair]})")
        
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
    
    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """Encode text to token IDs."""
        ids = []
        if add_special:
            ids.append(self.SPECIAL["<bos>"])
        for c in text:
            ids.append(self.char_to_id.get(c, self.SPECIAL["<unk>"]))
        if add_special:
            ids.append(self.SPECIAL["<eos>"])
        return ids
    
    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs to text."""
        special_ids = set(self.SPECIAL.values())
        return "".join([self.id_to_char.get(i, "") for i in ids
                       if not skip_special or i not in special_ids])
    
    def save(self, path: str):
        """Save tokenizer to file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "char_to_id": self.char_to_id,
                "merges": [list(m) for m in self.merges],
                "vocab_size": self.vocab_size,
            }, f)
    
    def load(self, path: str):
        """Load tokenizer from file."""
        with open(path) as f:
            data = json.load(f)
        self.char_to_id = data["char_to_id"]
        self.id_to_char = {int(v) if isinstance(v, str) and v.isdigit() else v: k for k, v in self.char_to_id.items()}
        self.merges = [tuple(m) for m in data.get("merges", [])]
        self.vocab_size = data.get("vocab_size", len(self.char_to_id))
    
    def __len__(self) -> int:
        return len(self.char_to_id)
