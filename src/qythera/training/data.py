"""Data pipeline. Pure Python + NumPy."""
import glob as _glob
import html
import json
import math
import mmap
import os
import re
import struct
import sys
import random
from collections import Counter, defaultdict
import functools
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np

from qythera.tensor import Tensor


# ---------------------------------------------------------------------------
# MMapDataset – memory-mapped binary dataset
# ---------------------------------------------------------------------------

class MMapDataset:
    def __init__(self, bin_path: str, meta_path: str, seq_len: int):
        self.seq_len = seq_len
        with open(meta_path) as f:
            meta = json.load(f)
        self.total_tokens = meta["total_tokens"]
        self.dtype = np.dtype(meta.get("dtype", "uint16"))
        self.item_size = self.dtype.itemsize
        self.num_sequences = max(0, (self.total_tokens - 1) // seq_len)
        self._file = open(bin_path, "rb")
        file_size = os.path.getsize(bin_path)
        if sys.platform == 'win32':
            self._mm = mmap.mmap(self._file.fileno(), file_size, access=mmap.ACCESS_READ)
        else:
            self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0:
            idx += self.num_sequences
        if idx < 0 or idx >= self.num_sequences:
            raise IndexError(f"index {idx} out of range [0, {self.num_sequences})")
        offset = idx * self.seq_len * self.item_size
        end = offset + self.seq_len * self.item_size
        buf = self._mm[offset:end]
        return np.frombuffer(buf, dtype=self.dtype).copy()

    def close(self):
        self._mm.close()
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


# ---------------------------------------------------------------------------
# StreamingDataset – JSONL streaming with token buffer
# ---------------------------------------------------------------------------

class StreamingDataset:
    def __init__(self, jsonl_path: str, tokenize_fn, seq_len: int, eos_id: int = 0):
        self.jsonl_path = jsonl_path
        self.tokenize_fn = tokenize_fn
        self.seq_len = seq_len
        self.eos_id = eos_id

    def __iter__(self) -> Iterator[List[int]]:
        buf: List[int] = []
        with open(self.jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj if isinstance(obj, str) else obj.get("text", "")
                tokens = self.tokenize_fn(text)
                buf.extend(tokens)
                buf.append(self.eos_id)
                while len(buf) >= self.seq_len:
                    chunk = buf[: self.seq_len]
                    buf = buf[self.seq_len:]
                    yield chunk


# ---------------------------------------------------------------------------
# DataLoader – shuffle, collate, pad, truncate
# ---------------------------------------------------------------------------

class DataLoader:
    def __init__(self, dataset, batch_size: int, seq_len: int, pad_id: int = 0,
                 shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.shuffle = shuffle

    def _fisher_yates(self, arr: list) -> list:
        a = list(arr)
        for i in range(len(a) - 1, 0, -1):
            j = int(np.random.randint(0, i + 1))
            a[i], a[j] = a[j], a[i]
        return a

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            indices = self._fisher_yates(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            batch = []
            for idx in batch_idx:
                item = self.dataset[idx]
                if isinstance(item, tuple) and len(item) == 2:
                    inp, tgt = item
                    if isinstance(inp, np.ndarray) and isinstance(tgt, np.ndarray):
                        batch.append({"input": inp, "target": tgt})
                        continue
                seq = item if isinstance(item, np.ndarray) else np.array(item)
                if len(seq) < self.seq_len:
                    pad = [self.pad_id] * (self.seq_len - len(seq))
                    seq = np.concatenate([seq, np.array(pad, dtype=seq.dtype)])
                elif len(seq) > self.seq_len:
                    seq = seq[: self.seq_len]
                batch.append(seq)
            if batch and isinstance(batch[0], dict):
                max_len = max(len(b["input"]) for b in batch)
                for b in batch:
                    if len(b["input"]) < max_len:
                        pad = [self.pad_id] * (max_len - len(b["input"]))
                        b["input"] = np.concatenate([b["input"], np.array(pad, dtype=b["input"].dtype)])
                        b["target"] = np.concatenate([b["target"], np.array(pad, dtype=b["target"].dtype)])
                yield {
                    "input": Tensor(np.stack([b["input"] for b in batch])),
                    "target": Tensor(np.stack([b["target"] for b in batch])),
                }
            else:
                yield Tensor(np.stack(batch))

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)


# ---------------------------------------------------------------------------
# DocumentPacker – multiple docs per sequence, separated by EOS
# ---------------------------------------------------------------------------

class DocumentPacker:
    def __init__(self, seq_len: int, eos_id: int = 0, pad_id: int = 0):
        self.seq_len = seq_len
        self.eos_id = eos_id
        self.pad_id = pad_id

    def pack(self, token_lists: List[List[int]]) -> List[List[int]]:
        sequences: List[List[int]] = []
        buf: List[int] = []
        for doc in token_lists:
            if len(buf) + len(doc) + 1 <= self.seq_len:
                buf.extend(doc)
                buf.append(self.eos_id)
            else:
                remaining = self.seq_len - len(buf)
                if remaining > 0:
                    buf.extend(doc[:remaining])
                while len(buf) < self.seq_len:
                    buf.append(self.pad_id)
                sequences.append(buf)
                buf = []
                if len(doc) >= self.seq_len:
                    chunk = doc[: self.seq_len]
                    sequences.append(chunk)
                    leftover = doc[self.seq_len:]
                    if leftover:
                        buf = leftover
                        buf.append(self.eos_id)
                else:
                    buf = list(doc)
                    buf.append(self.eos_id)
        if buf:
            while len(buf) < self.seq_len:
                buf.append(self.pad_id)
            sequences.append(buf)
        return sequences


# ---------------------------------------------------------------------------
# MinHashDeduplicator – k=9 hash functions, b=20 bands, Union-Find
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


class MinHashDeduplicator:
    def __init__(self, num_hashes: int = 9, num_bands: int = 20, seed: int = 42):
        self.k = num_hashes
        self.b = num_bands
        self.rows_per_band = max(1, num_hashes // num_bands)
        rng = np.random.RandomState(seed)
        self._primes = self._get_primes(num_hashes)
        self._a = rng.randint(1, 2**31 - 1, size=num_hashes).astype(np.uint64)
        self._b = rng.randint(0, 2**31 - 1, size=num_hashes).astype(np.uint64)
        self._mod = 2**61 - 1

    @staticmethod
    def _get_primes(n: int) -> np.ndarray:
        primes = []
        num = 2
        while len(primes) < n:
            if all(num % p != 0 for p in primes):
                primes.append(num)
            num += 1
        return np.array(primes, dtype=np.uint64)

    def _shingle(self, text: str, shingle_size: int = 3) -> set:
        tokens = text.lower().split()
        return {tuple(tokens[i:i + shingle_size]) for i in range(max(0, len(tokens) - shingle_size + 1))}

    def _minhash_signature(self, shingles: set) -> np.ndarray:
        sig = np.full(self.k, np.iinfo(np.uint64).max, dtype=np.uint64)
        for sh in shingles:
            h = sum(hash(s) for s in sh)
            for i in range(self.k):
                val = (self._a[i] * h + self._b[i]) % self._mod
                if val < sig[i]:
                    sig[i] = val
        return sig

    def _band_hashes(self, sig: np.ndarray) -> List[Tuple]:
        bands = []
        for b in range(self.b):
            start = b * self.rows_per_band
            end = start + self.rows_per_band
            band_vals = tuple(sig[start:end])
            bands.append((b, hash(band_vals)))
        return bands

    def deduplicate(self, texts: List[str]) -> List[str]:
        n = len(texts)
        uf = UnionFind(n)
        sigs = []
        band_maps = [defaultdict(list) for _ in range(self.b)]

        for i, text in enumerate(texts):
            shingles = self._shingle(text)
            sig = self._minhash_signature(shingles)
            sigs.append(sig)
            for b_idx, b_hash in self._band_hashes(sig):
                for j in band_maps[b_idx][b_hash]:
                    if uf.find(i) != uf.find(j):
                        si, sj = sigs[i], sigs[j]
                        common = np.sum(si == sj)
                        if common / self.k >= 0.8:
                            uf.union(i, j)
                band_maps[b_idx][b_hash].append(i)

        groups = defaultdict(list)
        for i in range(n):
            groups[uf.find(i)].append(i)
        return [texts[indices[0]] for indices in groups.values()]


# ---------------------------------------------------------------------------
# SimHash – 64 random hyperplanes, hamming distance
# ---------------------------------------------------------------------------

class SimHash:
    def __init__(self, dim: int = 64, seed: int = 42):
        self.dim = dim
        rng = np.random.RandomState(seed)
        self._seed = seed

    def _feature_vec(self, text: str) -> np.ndarray:
        tokens = text.lower().split()
        vocab = sorted(set(tokens))
        vec = np.zeros(len(vocab), dtype=np.float64)
        for t in tokens:
            vec[vocab.index(t)] += 1.0
        if np.linalg.norm(vec) > 0:
            vec /= np.linalg.norm(vec)
        return vec

    def hash_text(self, text: str) -> np.ndarray:
        vec = self._feature_vec(text)
        rng = np.random.RandomState(self._seed)
        projection_matrix = rng.randn(self.dim, len(vec)).astype(np.float64)
        projections = projection_matrix @ vec
        return (projections >= 0).astype(np.uint8)

    def hamming_distance(self, h1: np.ndarray, h2: np.ndarray) -> int:
        return int(np.sum(h1 != h2))

    def is_similar(self, h1: np.ndarray, h2: np.ndarray, threshold: int = 10) -> bool:
        return self.hamming_distance(h1, h2) <= threshold


# ---------------------------------------------------------------------------
# TextCleaner – HTML cleaning, entity decoding, PII removal
# ---------------------------------------------------------------------------

class TextCleaner:
    _HTML_TAG_RE = re.compile(r"<[^>]+>")
    _ENTITY_MAP = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
                   "&apos;": "'", "&#39;": "'", "&nbsp;": " ",
                   "&mdash;": "—", "&ndash;": "–", "&rsquo;": "\u2019",
                   "&lsquo;": "\u2018", "&rdquo;": "\u201d", "&ldquo;": "\u201c"}
    _EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    _PHONE_RE = re.compile(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
    _SSN_RE = re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b")
    _IP_RE = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
    _MULTI_SPACE = re.compile(r"\s+")

    def clean(self, text: str) -> str:
        text = self._HTML_TAG_RE.sub(" ", text)
        for ent, char in self._ENTITY_MAP.items():
            text = text.replace(ent, char)
        text = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), text)
        text = self._EMAIL_RE.sub("[EMAIL]", text)
        text = self._PHONE_RE.sub("[PHONE]", text)
        text = self._SSN_RE.sub("[SSN]", text)
        text = self._IP_RE.sub("[IP]", text)
        text = self._MULTI_SPACE.sub(" ", text).strip()
        return text


# ---------------------------------------------------------------------------
# TokenDataset – wraps tokenizer, produces (input, target) pairs
# ---------------------------------------------------------------------------

class TokenDataset:
    def __init__(self, data_source, tokenize_fn, seq_len: int, eos_id: int = 0):
        self.seq_len = seq_len
        self.eos_id = eos_id
        if isinstance(data_source, str):
            if data_source.endswith(".jsonl"):
                self._tokens = []
                with open(data_source) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        text = obj if isinstance(obj, str) else obj.get("text", "")
                        self._tokens.extend(tokenize_fn(text))
                        self._tokens.append(eos_id)
            else:
                with open(data_source) as f:
                    text = f.read()
                self._tokens = tokenize_fn(text)
        else:
            all_tokens: List[int] = []
            for item in data_source:
                text = item if isinstance(item, str) else item.get("text", "")
                all_tokens.extend(tokenize_fn(text))
                all_tokens.append(eos_id)
            self._tokens = all_tokens

        self.num_sequences = max(0, (len(self._tokens) - 1) // seq_len)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self._tokens[start:end]
        if len(chunk) < self.seq_len + 1:
            chunk = chunk + [self.eos_id] * (self.seq_len + 1 - len(chunk))
        inp = np.array(chunk[: self.seq_len], dtype=np.int64)
        tgt = np.array(chunk[1: self.seq_len + 1], dtype=np.int64)
        return inp, tgt


# ---------------------------------------------------------------------------
# PerplexityFilter – remove outlier documents by n-gram perplexity
# ---------------------------------------------------------------------------

class PerplexityFilter:
    def __init__(self, ngram: int = 5, lower_percentile: float = 5,
                 upper_percentile: float = 95):
        self.ngram = ngram
        self.lower = lower_percentile
        self.upper = upper_percentile

    def filter(self, documents: List[str]) -> List[str]:
        scores = [self._score(doc) for doc in documents]
        lo, hi = np.percentile(scores, [self.lower, self.upper])
        return [doc for doc, s in zip(documents, scores) if lo <= s <= hi]

    def _score(self, doc: str) -> float:
        tokens = doc.lower().split()
        if len(tokens) < self.ngram:
            return 0.0
        ngrams = Counter()
        for i in range(len(tokens) - self.ngram + 1):
            ng = tuple(tokens[i:i + self.ngram])
            ngrams[ng] += 1
        total = sum(ngrams.values())
        if total == 0:
            return 0.0
        log_prob = 0.0
        for count in ngrams.values():
            log_prob += count * math.log(count / total)
        return -log_prob / total


# ---------------------------------------------------------------------------
# LanguageIdentifier – character 3-gram language model
# ---------------------------------------------------------------------------

class LanguageIdentifier:
    def __init__(self):
        self.models: dict = {}

    def add_language(self, lang: str, texts: List[str]):
        counts: Counter = Counter()
        for text in texts:
            lower = text.lower()
            for i in range(len(lower) - 2):
                counts[lower[i:i + 3]] += 1
        total = sum(counts.values())
        self.models[lang] = {ng: c / total for ng, c in counts.items()}

    def identify(self, text: str) -> str:
        if not self.models:
            return ""
        lower = text.lower()
        return max(self.models.keys(), key=lambda l: self._loglik(lower, l))

    def _loglik(self, text: str, lang: str) -> float:
        model = self.models[lang]
        log_prob = 0.0
        for i in range(len(text) - 2):
            ng = text[i:i + 3]
            log_prob += math.log(model.get(ng, 1e-12))
        return log_prob


# ---------------------------------------------------------------------------
# HTMLCleaner – strip tags, decode entities, remove scripts/styles
# ---------------------------------------------------------------------------

class HTMLCleaner:
    def clean(self, html_text: str) -> str:
        text = re.sub(r'<script[^>]*>.*?</script>', '', html_text, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = html.unescape(text)
        return text.strip()


# ---------------------------------------------------------------------------
# ExactDeduplicator – suffix array, find and remove duplicate substrings
# ---------------------------------------------------------------------------

class ExactDeduplicator:
    def __init__(self, min_length: int = 50):
        self.min_length = min_length

    def deduplicate(self, documents: List[str]) -> List[str]:
        if not documents:
            return []
        if len(documents) == 1:
            return list(documents)

        sep = "\x00"
        concat = sep.join(documents)
        n = len(concat)

        sa = list(range(n))
        rank = [ord(concat[i]) for i in range(n)]
        k = 1
        tmp = [0] * n
        while True:
            def cmp_key(a):
                return (rank[a], rank[a + k] if a + k < n else -1)
            sa.sort(key=cmp_key)
            tmp[sa[0]] = 0
            for i in range(1, n):
                tmp[sa[i]] = tmp[sa[i - 1]]
                if cmp_key(sa[i]) != cmp_key(sa[i - 1]):
                    tmp[sa[i]] += 1
            rank, tmp = tmp, rank
            if rank[sa[-1]] == n - 1:
                break
            k <<= 1

        inv_sa = [0] * n
        for i, s in enumerate(sa):
            inv_sa[s] = i

        lcp = [0] * n
        h = 0
        for i in range(n):
            if inv_sa[i] > 0:
                j = sa[inv_sa[i] - 1]
                while i + h < n and j + h < n and concat[i + h] == concat[j + h]:
                    h += 1
                lcp[inv_sa[i]] = h
                if h > 0:
                    h -= 1

        doc_id = []
        cur = 0
        for ch in concat:
            if ch == sep:
                cur += 1
            doc_id.append(cur)

        uf = UnionFind(len(documents))
        for i in range(1, n):
            if lcp[i] >= self.min_length:
                d1 = doc_id[sa[i - 1]]
                d2 = doc_id[sa[i]]
                if d1 != d2:
                    uf.union(d1, d2)

        groups = defaultdict(list)
        for i in range(len(documents)):
            groups[uf.find(i)].append(i)
        return [documents[indices[0]] for indices in groups.values()]


# ---------------------------------------------------------------------------
# CurriculumScheduler – sort by difficulty, start with easiest
# ---------------------------------------------------------------------------

class CurriculumScheduler:
    def __init__(self, strategy: str = "easy_first"):
        self.strategy = strategy

    def sort(self, dataset: list) -> list:
        scored = [(self._score(x), i, x) for i, x in enumerate(dataset)]
        if self.strategy == "easy_first":
            scored.sort(key=lambda t: (t[0], t[1]))
        elif self.strategy == "hard_first":
            scored.sort(key=lambda t: (-t[0], t[1]))
        else:
            raise ValueError(f"unknown strategy: {self.strategy}")
        return [t[2] for t in scored]

    def _score(self, x):
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, dict):
            return float(x.get("perplexity", 0))
        if hasattr(x, "perplexity"):
            return float(x.perplexity)
        return float(x)


# ---------------------------------------------------------------------------
# ImportanceSampler – weight examples by inverse frequency
# ---------------------------------------------------------------------------

class ImportanceSampler:
    def __init__(self, frequencies):
        self.weights = np.array(1.0 / np.asarray(frequencies, dtype=np.float64))
        self.weights /= self.weights.sum()

    def sample(self, batch_size: int) -> np.ndarray:
        return np.random.choice(len(self.weights), batch_size, p=self.weights)


# ---------------------------------------------------------------------------
# TextDataset – read text file, tokenize, create (input, target) pairs
# ---------------------------------------------------------------------------

class TextDataset:
    def __init__(self, path: str, tokenizer, max_seq_len: int = 512, pad_id: int = 0,
                 eos_id: int = 0):
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        with open(path, "r") as f:
            text = f.read()
        tokens = tokenizer.encode(text)
        if not tokens or tokens[-1] != eos_id:
            tokens.append(eos_id)
        self._tokens = tokens
        self._num_samples = max(0, (len(tokens) - 1) // max_seq_len)

    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        start = idx * self.max_seq_len
        end = start + self.max_seq_len + 1
        chunk = self._tokens[start:end]
        if len(chunk) < self.max_seq_len + 1:
            chunk = chunk + [self.pad_id] * (self.max_seq_len + 1 - len(chunk))
        inp = np.array(chunk[: self.max_seq_len], dtype=np.int64)
        tgt = np.array(chunk[1: self.max_seq_len + 1], dtype=np.int64)
        return inp, tgt


# ---------------------------------------------------------------------------
# collate_fn – pad sequences in a batch to same length
# ---------------------------------------------------------------------------

def collate_fn(batch: list, pad_id: int = 0) -> dict:
    if not batch:
        return {"input": np.array([], dtype=np.int64), "target": np.array([], dtype=np.int64)}
    first = batch[0]
    if isinstance(first, dict):
        keys = list(first.keys())
        max_len = max(len(b[k]) for b in batch for k in keys)
        result = {}
        for k in keys:
            padded = []
            for b in batch:
                arr = np.array(b[k], dtype=np.int64)
                if len(arr) < max_len:
                    arr = np.concatenate([arr, np.full(max_len - len(arr), pad_id, dtype=np.int64)])
                padded.append(arr)
            result[k] = np.stack(padded)
        return result
    if isinstance(first, tuple) and len(first) == 2:
        max_len = max(len(b[0]) for b in batch)
        inputs, targets = [], []
        for inp, tgt in batch:
            inp = np.array(inp, dtype=np.int64)
            tgt = np.array(tgt, dtype=np.int64)
            if len(inp) < max_len:
                inp = np.concatenate([inp, np.full(max_len - len(inp), pad_id, dtype=np.int64)])
                tgt = np.concatenate([tgt, np.full(max_len - len(tgt), pad_id, dtype=np.int64)])
            inputs.append(inp)
            targets.append(tgt)
        return {"input": np.stack(inputs), "target": np.stack(targets)}
    max_len = max(len(b) for b in batch)
    padded = []
    for b in batch:
        arr = np.array(b, dtype=np.int64)
        if len(arr) < max_len:
            arr = np.concatenate([arr, np.full(max_len - len(arr), pad_id, dtype=np.int64)])
        padded.append(arr)
    return {"input": np.stack(padded), "target": np.stack(padded)}


# ---------------------------------------------------------------------------
# SimpleTextCorpus – load multiple text files, merge, shuffle
# ---------------------------------------------------------------------------

class SimpleTextCorpus:
    def __init__(self, paths: Optional[List[str]] = None, pattern: Optional[str] = None,
                 shuffle: bool = True, seed: int = 42):
        self.shuffle = shuffle
        self.seed = seed
        if paths is None and pattern is not None:
            paths = sorted(_glob.glob(pattern))
        if not paths:
            raise ValueError("no files provided")
        self._files = list(paths)
        self._docs = []
        for p in self._files:
            with open(p, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._docs.append(line)
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self._docs)

    def __len__(self):
        return len(self._docs)

    def __iter__(self):
        return iter(self._docs)

    def texts(self) -> List[str]:
        return list(self._docs)


# ---------------------------------------------------------------------------
# SampleTrainingDataGenerator – synthetic training data for testing
# ---------------------------------------------------------------------------

class SampleTrainingDataGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate(self, path: str, num_samples: int = 100, min_len: int = 10, max_len: int = 50):
        vocab = list("abcdefghijklmnopqrstuvwxyz ")
        with open(path, "w") as f:
            for _ in range(num_samples):
                length = self.rng.randint(min_len, max_len)
                line = "".join(self.rng.choices(vocab, k=length))
                f.write(line + "\n")

    def generate_corpus(self, path: str, num_files: int = 3, samples_per_file: int = 50,
                        min_len: int = 10, max_len: int = 50) -> List[str]:
        paths = []
        for i in range(num_files):
            fp = f"{path}_{i}.txt"
            self.generate(fp, num_samples=samples_per_file, min_len=min_len, max_len=max_len)
            paths.append(fp)
        return paths
