"""Data pipeline. Pure Python + NumPy."""
import json
import math
import mmap
import os
import re
import struct
from collections import defaultdict
from typing import Iterator, List, Optional, Tuple

import numpy as np

from core.tensor import Tensor


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
                seq = self.dataset[idx]
                if len(seq) < self.seq_len:
                    pad = [self.pad_id] * (self.seq_len - len(seq))
                    seq = np.concatenate([seq, np.array(pad, dtype=seq.dtype)])
                elif len(seq) > self.seq_len:
                    seq = seq[: self.seq_len]
                batch.append(seq)
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
