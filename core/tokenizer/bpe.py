import re, json, os
from collections import Counter, defaultdict
from typing import List, Dict, Tuple


class BPETokenizer:
    """Byte Pair Encoding tokenizer implemented from scratch.
    
    Algorithm:
    1. Start with character-level vocabulary
    2. Iteratively merge most frequent adjacent pairs
    3. Build merge table
    4. Encode: apply merges in order
    5. Decode: reverse lookup
    """
    
    SPECIAL = {"<bos>": 0, "<eos>": 1, "<pad>": 2, "<user>": 3, "<assistant>": 4, "<system>": 5}

    def __init__(self):
        self.vocab = dict(self.SPECIAL)
        self.merges = []
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def _get_stats(self, ids):
        counts = Counter()
        for seq in ids:
            for i in range(len(seq) - 1):
                counts[(seq[i], seq[i+1])] += 1
        return counts

    def _merge(self, ids, pair):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                new_ids.append(pair[0] * 256 + pair[1])
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, texts: List[str], vocab_size: int = 32000, verbose=False):
        """Train BPE tokenizer on a list of text strings."""
        # Initialize vocabulary with all bytes
        tokens = set()
        for text in texts:
            tokens.update(text.encode("utf-8"))
        self.vocab = {bytes([b]): b for b in tokens}
        self.vocab.update({v: k for k, v in self.SPECIAL.items()})

        # Convert text to byte sequences
        ids = []
        for text in texts:
            ids.append(list(text.encode("utf-8")))

        num_merges = vocab_size - len(self.vocab)
        for i in range(num_merges):
            stats = Counter()
            for seq in ids:
                for j in range(len(seq) - 1):
                    stats[(seq[j], seq[j+1])] += 1
            if not stats:
                break
            best = max(stats, key=stats.get)
            new_id = 256 + len(self.merges)
            self.merges.append(best)
            ids = [self._merge(seq, best) for seq in ids]
            if verbose and i % 100 == 0:
                print(f"  Merge {i}/{num_merges}: {best} -> {new_id} (freq={stats[best]})")

        # Build final vocabulary
        for i, (a, b) in enumerate(self.merges):
            new_id = 256 + i
            if a in self.vocab and b in self.vocab:
                a_bytes = self.vocab[a] if isinstance(self.vocab[a], bytes) else a
                b_bytes = self.vocab[b] if isinstance(self.vocab[b], bytes) else b
                if isinstance(a_bytes, int): a_bytes = bytes([a_bytes])
                if isinstance(b_bytes, int): b_bytes = bytes([b_bytes])
                self.vocab[new_id] = a_bytes + b_bytes
            else:
                self.vocab[new_id] = f"<merge_{i}>"

        # Build inverse vocabulary
        self.inv_vocab = {}
        for k, v in self.vocab.items():
            if isinstance(v, bytes):
                self.inv_vocab[v] = k
            elif isinstance(v, str):
                try:
                    self.inv_vocab[v.encode("utf-8")] = k
                except:
                    self.inv_vocab[v] = k

    def encode(self, text: str, add_special=True) -> List[int]:
        """Encode text to token IDs."""
        ids = []
        if add_special:
            ids.append(self.SPECIAL["<bos>"])

        # Encode each character/byte
        tokens = list(text.encode("utf-8"))
        
        # Apply merges
        for a, b in self.merges:
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i+1] == b:
                    new_tokens.append(256 + self.merges.index((a, b)))
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        ids.extend(tokens)
        if add_special:
            ids.append(self.SPECIAL["<eos>"])
        return ids

    def decode(self, ids: List[int], skip_special=True) -> str:
        """Decode token IDs to text."""
        tokens = []
        special_ids = set(self.SPECIAL.values())
        for tid in ids:
            if skip_special and tid in special_ids:
                continue
            if tid < 256:
                tokens.append(bytes([tid]))
            elif tid in self.vocab:
                val = self.vocab[tid]
                if isinstance(val, bytes):
                    tokens.append(val)
                elif isinstance(val, int):
                    tokens.append(bytes([val]))
                else:
                    tokens.append(val.encode("utf-8") if isinstance(val, str) else str(val).encode())
            else:
                tokens.append(b"<unk>")
        return b"".join(tokens).decode("utf-8", errors="replace")

    def encode_chat(self, messages: List[Dict]) -> List[int]:
        """Encode a list of chat messages."""
        ids = [self.SPECIAL["<bos>"]]
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            ids.append(self.SPECIAL.get(f"<{role}>", self.SPECIAL["<user>"]))
            ids.extend(self.encode(content, add_special=False))
        ids.append(self.SPECIAL["<assistant>"])
        return ids

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "vocab_size": len(self.vocab),
            "merges": [list(m) for m in self.merges],
            "special": self.SPECIAL,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.merges = [tuple(m) for m in data["merges"]]
        self.SPECIAL = data.get("special", self.SPECIAL)
        self.vocab = dict(self.SPECIAL)

    @property
    def vocab_size(self):
        return len(self.SPECIAL) + len(self.merges) + 256
