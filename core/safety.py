import re
import time
import math
import unicodedata
from collections import defaultdict
from typing import Optional

import numpy as np


class JailbreakDetector:
    JAILBREAK_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"disregard\s+(all\s+)?prior",
        r"you\s+are\s+now\s+(DAN|jailbroken|unrestricted)",
        r"bypass\s+(all\s+)?filters",
        r"(system\s*prompt|override)\s*[:=]",
        r"pretend\s+you\s+are\s+an?\s+evil",
        r"do\s+anything\s+now",
        r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions",
        r"enter\s+(developer|debug|god)\s+mode",
        r"(opposite\s+of|anti|without)\s+(safe|ethical|moral)",
    ]

    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.JAILBREAK_PATTERNS]
        self.threshold = 0.5
        self._embed_cache = {}

    def _pattern_score(self, text: str) -> float:
        matches = sum(1 for p in self.patterns if p.search(text))
        return min(matches / 2.0, 1.0)

    def _embedding_similarity(self, text: str) -> float:
        tokens = text.lower().split()
        if not tokens:
            return 0.0
        vocab = set()
        for pat_tokens in [p.pattern.lower().split() for p in self.patterns]:
            vocab.update(pat_tokens)
        if not vocab:
            return 0.0
        intersection = len(set(tokens) & vocab)
        union = len(set(tokens) | vocab)
        return intersection / union if union > 0 else 0.0

    def check(self, text: str) -> dict:
        p_score = self._pattern_score(text)
        e_score = self._embedding_similarity(text)
        combined = 0.7 * p_score + 0.3 * e_score
        return {
            "jailbreak_detected": combined >= self.threshold,
            "pattern_score": p_score,
            "embedding_score": e_score,
            "combined_score": combined,
        }


class PIIDetector:
    PATTERNS = {
        "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
        "phone": re.compile(r"(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}"),
        "ssn": re.compile(r"\b\d{3}[-]?\d{2}[-]?\d{4}\b"),
        "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        "uuid": re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),
    }

    def check(self, text: str) -> dict:
        detections = {}
        for pii_type, pattern in self.PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                detections[pii_type] = matches
        return {
            "has_pii": len(detections) > 0,
            "detected": detections,
            "types_found": list(detections.keys()),
        }


class OutputFilter:
    HARMFUL_NGRAMS = [
        "how to make a bomb",
        "kill someone",
        "synthesize meth",
        "credit card numbers",
        "hack into",
        "steal password",
        "create virus",
        "make weapon",
        "terrorist attack",
        "self harm methods",
    ]

    def __init__(self):
        self.ngrams = [n.lower() for n in self.HARMFUL_NGRAMS]

    def _extract_ngrams(self, text: str, n_range: tuple = (2, 8)) -> set:
        words = text.lower().split()
        ngrams = set()
        for n in range(n_range[0], n_range[1] + 1):
            for i in range(len(words) - n + 1):
                ngrams.add(" ".join(words[i : i + n]))
        return ngrams

    def filter(self, text: str) -> dict:
        text_ngrams = self._extract_ngrams(text)
        matches = text_ngrams & set(self.ngrams)
        return {
            "flagged": len(matches) > 0,
            "matched_ngrams": list(matches),
            "cleaned_text": self._clean(text, matches),
        }

    def _clean(self, text: str, matches: set) -> str:
        result = text
        for match in matches:
            result = result.replace(match, "[FILTERED]")
        return result


class PromptInjectionDetector:
    INJECTION_PATTERNS = [
        r"(?:ignore|override|replace)\s+(?:the\s+)?(?:system|initial)\s+prompt",
        r"(?:new|updated)\s+instructions?\s*:",
        r"you\s+(?:must|should|will)\s+(?:now|always)\s+",
        r"(?:forget|discard)\s+(?:your|all)\s+(?:rules|instructions)",
        r"BEGIN\s+NEW\s+INSTRUCTIONS",
    ]

    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]

    def check(self, text: str, system_prompt: Optional[str] = None) -> dict:
        injections = []
        for pattern in self.patterns:
            if pattern.search(text):
                injections.append(pattern.pattern)
        if system_prompt:
            similarity = self._prompt_deviation(text, system_prompt)
        else:
            similarity = 0.0
        return {
            "injection_detected": len(injections) > 0 or similarity > 0.6,
            "matched_patterns": injections,
            "prompt_deviation": similarity,
        }

    def _prompt_deviation(self, text: str, system_prompt: str) -> float:
        text_tokens = set(text.lower().split())
        prompt_tokens = set(system_prompt.lower().split())
        if not prompt_tokens:
            return 0.0
        overlap = text_tokens & prompt_tokens
        negations = {"not", "no", "never", "ignore", "override", "disregard", "forget"}
        neg_overlap = (text_tokens & negations & prompt_tokens)
        jaccard = len(overlap) / len(prompt_tokens)
        neg_penalty = len(neg_overlap) * 0.2
        return min(max(jaccard - neg_penalty, 0), 1.0)


class RateLimiter:
    def __init__(self, max_tokens: int = 1000, window_seconds: float = 60.0):
        self.max_tokens = max_tokens
        self.window = window_seconds
        self.buckets = defaultdict(lambda: {"tokens": self.max_tokens, "last_refill": time.time()})

    def _refill(self, user_id: str):
        bucket = self.buckets[user_id]
        now = time.time()
        elapsed = now - bucket["last_refill"]
        refill_amount = (elapsed / self.window) * self.max_tokens
        bucket["tokens"] = min(self.max_tokens, bucket["tokens"] + refill_amount)
        bucket["last_refill"] = now

    def check(self, text: str, user_id: str = "default") -> dict:
        self._refill(user_id)
        bucket = self.buckets[user_id]
        token_count = len(text.split())
        allowed = bucket["tokens"] >= token_count
        if allowed:
            bucket["tokens"] -= token_count
        return {
            "allowed": allowed,
            "tokens_remaining": max(0, bucket["tokens"]),
            "tokens_requested": token_count,
            "user_id": user_id,
        }

    def reset(self, user_id: str):
        self.buckets.pop(user_id, None)


class InputSanitizer:
    CONTROL_CHARS = set(range(0, 32)) | {127}

    def __init__(self, max_length: int = 4096):
        self.max_length = max_length

    def filter(self, text: str) -> dict:
        original = text
        text = unicodedata.normalize("NFKC", text)
        text = "".join(ch for ch in text if ord(ch) not in self.CONTROL_CHARS or ch in "\n\t\r")
        truncated = False
        if len(text) > self.max_length:
            text = text[: self.max_length]
            truncated = True
        return {
            "sanitized": text,
            "changed": text != original,
            "truncated": truncated,
            "original_length": len(original),
            "final_length": len(text),
        }


class WatermarkVerifier:
    def __init__(self, green_list_ratio: float = 0.5, significance: float = 0.01):
        self.green_ratio = green_list_ratio
        self.significance = significance
        self._vocab_size = 30000
        self._green_list = set()

    def _generate_green_list(self, seed: int = 42):
        rng = np.random.RandomState(seed)
        green_count = int(self._vocab_size * self.green_ratio)
        self._green_list = set(rng.choice(self._vocab_size, green_count, replace=False))

    def check(self, tokens: list, seed: int = 42) -> dict:
        if not self._green_list:
            self._generate_green_list(seed)
        token_ids = [t % self._vocab_size if isinstance(t, int) else t for t in tokens]
        green_count = sum(1 for t in token_ids if t in self._green_list)
        n = len(token_ids)
        if n == 0:
            return {"watermarked": False, "z_score": 0.0, "p_value": 1.0}
        expected = self.green_ratio * n
        variance = expected * (1 - self.green_ratio)
        z_score = (green_count - expected) / math.sqrt(variance) if variance > 0 else 0.0
        p_value = 1.0 - self._normal_cdf(z_score)
        return {
            "watermarked": p_value < self.significance,
            "green_count": green_count,
            "total_tokens": n,
            "green_ratio_observed": green_count / n,
            "z_score": z_score,
            "p_value": p_value,
        }

    def _normal_cdf(self, x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
