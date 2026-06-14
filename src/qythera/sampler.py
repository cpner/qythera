import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod


def apply_penalties(
    logits: np.ndarray,
    seen_tokens: List[int],
    repetition: float = 1.0,
    presence: float = 0.0,
    frequency: float = 0.0,
) -> np.ndarray:
    if not seen_tokens or (repetition == 1.0 and presence == 0.0 and frequency == 0.0):
        return logits
    logits = logits.copy()
    counts: Dict[int, int] = {}
    for t in seen_tokens:
        counts[t] = counts.get(t, 0) + 1
    for token, count in counts.items():
        if repetition != 1.0:
            logits[token] = logits[token] / repetition
        if presence != 0.0:
            logits[token] -= presence
        if frequency != 0.0:
            logits[token] -= frequency * count
    return logits


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = logits / temperature
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    logits = logits.copy()
    top_k_idx = np.argpartition(logits, -k)[-k:]
    mask = np.ones_like(logits, dtype=bool)
    mask[top_k_idx] = False
    logits[mask] = -np.inf
    return logits


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    sorted_indices = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]
    cum_probs = np.cumsum(softmax(sorted_logits))
    cutoff_idx = np.searchsorted(cum_probs, p)
    mask = np.ones_like(logits, dtype=bool)
    mask[sorted_indices[:cutoff_idx + 1]] = False
    logits = logits.copy()
    logits[mask] = -np.inf
    return logits


def typical_filter(logits: np.ndarray, p: float) -> np.ndarray:
    probs = softmax(logits)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    log_probs = np.log(probs + 1e-10)
    conditional_entropy = -(log_probs - entropy)
    threshold = np.percentile(conditional_entropy, (1 - p) * 100)
    mask = conditional_entropy > threshold
    logits = logits.copy()
    logits[mask] = -np.inf
    return logits


class Sampler(ABC):
    @abstractmethod
    def sample(self, logits: np.ndarray, **kwargs) -> int:
        pass


class GreedySampler(Sampler):
    def sample(self, logits: np.ndarray, **kwargs) -> int:
        return int(np.argmax(logits))


class TemperatureSampler(Sampler):
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def sample(self, logits: np.ndarray, **kwargs) -> int:
        probs = softmax(logits, self.temperature)
        return int(np.random.choice(len(logits), p=probs))


class TopKSampler(Sampler):
    def __init__(self, k: int = 50, temperature: float = 1.0):
        self.k = k
        self.temperature = temperature

    def sample(self, logits: np.ndarray, **kwargs) -> int:
        filtered = top_k_filter(logits, self.k)
        probs = softmax(filtered, self.temperature)
        return int(np.random.choice(len(logits), p=probs))


class TopPSampler(Sampler):
    def __init__(self, p: float = 0.9, temperature: float = 1.0):
        self.p = p
        self.temperature = temperature

    def sample(self, logits: np.ndarray, **kwargs) -> int:
        filtered = top_p_filter(logits, self.p)
        probs = softmax(filtered, self.temperature)
        return int(np.random.choice(len(logits), p=probs))


class MinPSampler(Sampler):
    def __init__(self, min_p: float = 0.05, temperature: float = 1.0):
        self.min_p = min_p
        self.temperature = temperature

    def sample(self, logits: np.ndarray, **kwargs) -> int:
        probs = softmax(logits, self.temperature)
        max_prob = np.max(probs)
        threshold = self.min_p * max_prob
        logits = logits.copy()
        logits[probs < threshold] = -np.inf
        filtered_probs = softmax(logits, self.temperature)
        return int(np.random.choice(len(logits), p=filtered_probs))


class TypicalSampler(Sampler):
    def __init__(self, p: float = 0.9, temperature: float = 1.0):
        self.p = p
        self.temperature = temperature

    def sample(self, logits: np.ndarray, **kwargs) -> int:
        filtered = typical_filter(logits, self.p)
        probs = softmax(filtered, self.temperature)
        return int(np.random.choice(len(logits), p=probs))


class RepetitionPenalty:
    def __init__(self, penalty: float = 1.2):
        self.penalty = penalty

    def __call__(self, logits: np.ndarray, seen_tokens: List[int]) -> np.ndarray:
        return apply_penalties(logits, seen_tokens, repetition=self.penalty)


class PresencePenalty:
    def __init__(self, penalty: float = 0.1):
        self.penalty = penalty

    def __call__(self, logits: np.ndarray, seen_tokens: List[int]) -> np.ndarray:
        return apply_penalties(logits, seen_tokens, presence=self.penalty)


class FrequencyPenalty:
    def __init__(self, penalty: float = 0.1):
        self.penalty = penalty

    def __call__(self, logits: np.ndarray, seen_tokens: List[int]) -> np.ndarray:
        return apply_penalties(logits, seen_tokens, frequency=self.penalty)


class ContrastiveDecoding:
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def decode(
        self, logits_large: np.ndarray, logits_small: np.ndarray
    ) -> np.ndarray:
        return logits_large - self.alpha * logits_small


class BeamSearch:
    def __init__(self, width: int = 4, length_penalty: float = 0.6):
        self.width = width
        self.length_penalty = length_penalty

    def search(
        self,
        logits_fn,
        start_token: int,
        max_length: int = 50,
        eos_token: int = 0,
    ) -> List[List[int]]:
        sequences: List[Tuple[List[int], float]] = [([start_token], 0.0)]
        for _ in range(max_length):
            all_candidates = []
            for seq, score in sequences:
                if seq[-1] == eos_token:
                    all_candidates.append((seq, score))
                    continue
                logits = logits_fn(seq)
                probs = softmax(logits)
                top_k_idx = np.argsort(probs)[-self.width :]
                for idx in top_k_idx:
                    new_seq = seq + [int(idx)]
                    new_score = score + np.log(probs[idx] + 1e-10)
                    all_candidates.append((new_seq, new_score))
            ranked = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = ranked[: self.width]
            if all(s[-1] == eos_token for s, _ in sequences):
                break
        results = []
        for seq, score in sequences:
            length = len(seq)
            normalized = score / (length ** self.length_penalty)
            results.append((seq, normalized))
        results.sort(key=lambda x: x[1], reverse=True)
        return [seq for seq, _ in results]


class SelfConsistency:
    def __init__(self, n_samples: int = 10, sampler: Optional[Sampler] = None):
        self.n_samples = n_samples
        self.sampler = sampler or TemperatureSampler(temperature=1.0)

    def generate(self, logits_fn, max_length: int = 50) -> List[List[int]]:
        samples = []
        for _ in range(self.n_samples):
            tokens = []
            for _ in range(max_length):
                logits = logits_fn(tokens)
                token = self.sampler.sample(logits)
                tokens.append(token)
            samples.append(tokens)
        return samples

    def majority_vote(self, samples: List[List[int]]) -> List[int]:
        if not samples:
            return []
        min_len = min(len(s) for s in samples)
        result = []
        for i in range(min_len):
            counts = {}
            for s in samples:
                token = s[i]
                counts[token] = counts.get(token, 0) + 1
            result.append(max(counts, key=counts.get))
        return result


class WatermarkDetector:
    def __init__(self, vocab_size: int, gamma: float = 0.5, seed: int = 42):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.rng = np.random.RandomState(seed)
        self.green_list = self._generate_green_list()

    def _generate_green_list(self) -> np.ndarray:
        indices = self.rng.choice(self.vocab_size, size=int(self.vocab_size * self.gamma), replace=False)
        mask = np.zeros(self.vocab_size, dtype=bool)
        mask[indices] = True
        return mask

    def is_green(self, token: int) -> bool:
        return self.green_list[token]

    def detect(self, tokens: List[int]) -> Tuple[float, bool]:
        if len(tokens) < 2:
            return 0.0, False
        context_tokens = tokens[:-1]
        green_count = sum(1 for t in context_tokens if self.is_green(t))
        n = len(context_tokens)
        p0 = self.gamma
        p_hat = green_count / n
        se = np.sqrt(p0 * (1 - p0) / n)
        z_score = (p_hat - p0) / se if se > 0 else 0.0
        is_watermarked = z_score > 1.96
        return float(z_score), bool(is_watermarked)


def sample_with_strategies(
    logits: np.ndarray,
    strategies: List[Any],
    seen_tokens: Optional[List[int]] = None,
) -> int:
    seen_tokens = seen_tokens or []
    for strategy in strategies:
        if isinstance(strategy, (RepetitionPenalty, PresencePenalty, FrequencyPenalty)):
            logits = strategy(logits, seen_tokens)
        elif isinstance(strategy, ContrastiveDecoding):
            pass
        elif isinstance(strategy, Sampler):
            return strategy.sample(logits)
    return int(np.argmax(logits))
