
import json, os
from typing import List, Dict

class Tokenizer:
    SPECIAL = {"bos":0, "eos":1, "pad":2, "user":3, "assistant":4, "system":5}
    def __init__(self, vocab_path=None):
        self.vocab = dict(self.SPECIAL)
        self.inv_vocab = {v:k for k,v in self.vocab.items()}
        if vocab_path and os.path.exists(vocab_path):
            with open(vocab_path) as f: self.vocab = json.load(f)
            self.inv_vocab = {v:k for k,v in self.vocab.items()}
        else:
            for i in range(32, 127):
                if chr(i) not in self.vocab:
                    self.vocab[chr(i)] = len(self.vocab)
                    self.inv_vocab[len(self.vocab)-1] = chr(i)
        self.vocab_size = len(self.vocab)

    def encode(self, text: str, add_special=True) -> List[int]:
        ids = [self.SPECIAL["bos"]] if add_special else []
        for ch in text:
            ids.append(self.vocab.get(ch, self.vocab.get("pad", 2)))
        if add_special: ids.append(self.SPECIAL["eos"])
        return ids

    def decode(self, ids: List[int], skip_special=True) -> str:
        chars = []
        special_ids = set(self.SPECIAL.values())
        for i in ids:
            if skip_special and i in special_ids: continue
            chars.append(self.inv_vocab.get(i, ""))
        return "".join(chars)

    def encode_chat(self, messages: List[Dict]) -> List[int]:
        ids = [self.SPECIAL["bos"]]
        for m in messages:
            role = m.get("role", "user")
            ids.append(self.SPECIAL.get(role, self.SPECIAL["user"]))
            ids.extend(self.encode(m.get("content",""), add_special=False))
        ids.append(self.SPECIAL["assistant"])
        return ids

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f: json.dump(self.vocab, f)
