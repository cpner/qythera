import re
import time
import math
import unicodedata
from collections import defaultdict
from typing import Optional

import numpy as np

from qythera.tensor import Tensor


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


class DPSGDOptimizer:
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_norm: float = 1.0,
        noise_multiplier: Optional[float] = None,
        batch_size: int = 32,
        dataset_size: int = 10000,
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.max_norm = max_norm
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        if noise_multiplier is not None:
            self.noise_multiplier = noise_multiplier
        else:
            self.noise_multiplier = max_norm * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
        self.steps = 0
        self.alpha = 1.0
        self.rdp_steps = 0.0

    def clip_gradients(self, gradients: list) -> np.ndarray:
        clipped = []
        for g in gradients:
            grad_norm = np.linalg.norm(g)
            if grad_norm > self.max_norm:
                g = g * self.max_norm / grad_norm
            clipped.append(g)
        return clipped

    def add_noise(self, clipped_gradients: list) -> np.ndarray:
        stacked = np.array(clipped_gradients)
        mean_grad = np.mean(stacked, axis=0)
        noise = np.random.normal(0, self.noise_multiplier * self.max_norm, size=mean_grad.shape)
        return mean_grad + noise

    def step(self, gradients: list) -> np.ndarray:
        clipped = self.clip_gradients(gradients)
        noisy_grad = self.add_noise(clipped)
        self.steps += 1
        self._update_rdp()
        return noisy_grad

    def _update_rdp(self):
        q = self.batch_size / self.dataset_size
        self.rdp_steps += q * self.noise_multiplier ** 2

    def get_privacy_spent(self) -> dict:
        epsilon_rdp = self.rdp_steps
        delta_converted = self.delta
        return {
            "epsilon": epsilon_rdp,
            "delta": delta_converted,
            "noise_multiplier": self.noise_multiplier,
            "steps": self.steps,
            "max_norm": self.max_norm,
        }

    def compose_epsilon(self, target_delta: Optional[float] = None) -> float:
        if target_delta is None:
            target_delta = self.delta
        alpha = self.alpha
        rdp = self.rdp_steps
        epsilon_rdp = rdp + math.log(1.0 / target_delta) / (alpha - 1)
        return epsilon_rdp


class AdversarialRobustness:
    def __init__(self, model, epsilon: float = 0.03, alpha: float = 0.007, num_steps: int = 10):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps

    def fgsm_attack(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_tensor = Tensor(x.astype(np.float32))
        y_tensor = Tensor(y.astype(np.int32))
        logits = self.model(x_tensor)
        loss = self._cross_entropy(logits, y_tensor)
        grad = self._compute_grad(loss, x_tensor)
        x_adv = x + self.epsilon * np.sign(grad)
        return np.clip(x_adv, x - self.epsilon, x + self.epsilon)

    def pgd_attack(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_adv = x.copy()
        for _ in range(self.num_steps):
            x_tensor = Tensor(x_adv.astype(np.float32))
            y_tensor = Tensor(y.astype(np.int32))
            logits = self.model(x_tensor)
            loss = self._cross_entropy(logits, y_tensor)
            grad = self._compute_grad(loss, x_tensor)
            x_adv = x_adv + self.alpha * np.sign(grad)
            perturbation = np.clip(x_adv - x, -self.epsilon, self.epsilon)
            x_adv = x + perturbation
        return x_adv

    def adversarial_training_step(self, x: np.ndarray, y: np.ndarray) -> float:
        x_adv = self.pgd_attack(x, y)
        x_tensor = Tensor(x_adv.astype(np.float32))
        y_tensor = Tensor(y.astype(np.int32))
        logits = self.model(x_tensor)
        loss = self._cross_entropy(logits, y_tensor)
        return float(loss.data)

    def _cross_entropy(self, logits: Tensor, targets: Tensor) -> Tensor:
        log_probs = logits.data - np.log(np.exp(logits.data).sum(axis=-1, keepdims=True) + 1e-8)
        B, L, V = log_probs.shape
        targets_flat = targets.data.flatten()
        token_log_probs = log_probs.reshape(-1, V)[np.arange(len(targets_flat)), targets_flat]
        return Tensor(-token_log_probs.mean().reshape(1))

    def _compute_grad(self, loss: Tensor, x: Tensor) -> np.ndarray:
        grad = np.random.randn(*x.data.shape) * 0.01
        return grad

    def evaluate_robustness(self, x_test: np.ndarray, y_test: np.ndarray, attacks: list = None) -> dict:
        if attacks is None:
            attacks = ["fgsm", "pgd"]
        results = {}
        clean_correct = 0
        total = len(x_test)
        for atk in attacks:
            correct = 0
            for i in range(total):
                if atk == "fgsm":
                    x_adv = self.fgsm_attack(x_test[i:i+1], y_test[i:i+1])
                else:
                    x_adv = self.pgd_attack(x_test[i:i+1], y_test[i:i+1])
                logits = self.model(Tensor(x_adv.astype(np.float32)))
                pred = np.argmax(logits.data[0, -1])
                if pred == y_test[i, -1]:
                    correct += 1
            results[f"{atk}_accuracy"] = correct / total
        return results


class MembershipInferenceDefense:
    def __init__(self, loss_noise_std: float = 0.1, calibration_bins: int = 10):
        self.loss_noise_std = loss_noise_std
        self.calibration_bins = calibration_bins
        self.calibration_map: dict = {}

    def add_loss_noise(self, losses: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.loss_noise_std, size=losses.shape)
        return losses + noise

    def compute_confidence(self, logits: np.ndarray) -> np.ndarray:
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)
        return probs.max(axis=-1)

    def calibrate(self, confidences: np.ndarray, labels: np.ndarray):
        bins = np.linspace(0, 1, self.calibration_bins + 1)
        for i in range(self.calibration_bins):
            mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            if mask.sum() > 0:
                self.calibration_map[i] = labels[mask].mean()
            else:
                self.calibration_map[i] = (bins[i] + bins[i+1]) / 2

    def calibrated_confidence(self, confidence: float) -> float:
        bins = np.linspace(0, 1, self.calibration_bins + 1)
        bin_idx = min(int(confidence * self.calibration_bins), self.calibration_bins - 1)
        return self.calibration_map.get(bin_idx, confidence)

    def defense_score(self, logits: np.ndarray, labels: np.ndarray) -> dict:
        confidences = self.compute_confidence(logits)
        noisy_losses = self.add_loss_noise(-np.log(confidences + 1e-8))
        calibrated = np.array([self.calibrated_confidence(c) for c in confidences])
        return {
            "confidences": confidences,
            "noisy_losses": noisy_losses,
            "calibrated_confidences": calibrated,
            "membership_score": float(np.mean(calibrated)),
        }


class BackdoorDetector:
    def __init__(self, num_clusters: int = 2, spectral_threshold: float = 0.5):
        self.num_clusters = num_clusters
        self.spectral_threshold = spectral_threshold
        self.clean_activations: Optional[np.ndarray] = None
        self.suspicious_activations: Optional[np.ndarray] = None

    def set_clean_activations(self, activations: np.ndarray):
        self.clean_activations = activations

    def set_suspicious_activations(self, activations: np.ndarray):
        self.suspicious_activations = activations

    def activation_clustering(self) -> dict:
        if self.clean_activations is None or self.suspicious_activations is None:
            return {"clustered": False, "reason": "activations not set"}
        clean_mean = np.mean(self.clean_activations, axis=0)
        suspicious_mean = np.mean(self.suspicious_activations, axis=0)
        clean_std = np.std(self.clean_activations, axis=0) + 1e-8
        z_scores = np.abs(suspicious_mean - clean_mean) / clean_std
        backdoor_detected = np.mean(z_scores > 3.0) > self.spectral_threshold
        return {
            "backdoor_detected": backdoor_detected,
            "mean_z_score": float(np.mean(z_scores)),
            "max_z_score": float(np.max(z_scores)),
            "fraction_outlier": float(np.mean(z_scores > 3.0)),
        }

    def spectral_signature(self) -> dict:
        if self.clean_activations is None or self.suspicious_activations is None:
            return {"detected": False, "reason": "activations not set"}
        combined = np.vstack([self.clean_activations, self.suspicious_activations])
        mean_centered = combined - combined.mean(axis=0)
        _, s, _ = np.linalg.svd(mean_centered, full_matrices=False)
        if len(s) < 2:
            return {"detected": False, "reason": "insufficient dimensions"}
        spectral_gap = s[0] - s[1]
        mean_s = np.mean(s)
        ratio = spectral_gap / (mean_s + 1e-8)
        suspicious_ratio = len(self.suspicious_activations) / len(combined)
        detected = ratio > self.spectral_threshold and suspicious_ratio < 0.5
        return {
            "detected": detected,
            "spectral_gap": float(spectral_gap),
            "singular_value_ratio": float(ratio),
            "suspicious_fraction": float(suspicious_ratio),
            "top_singular_values": s[:5].tolist(),
        }

    def detect(self) -> dict:
        cluster_result = self.activation_clustering()
        spectral_result = self.spectral_signature()
        backdoor_detected = cluster_result.get("backdoor_detected", False) or spectral_result.get("detected", False)
        return {
            "backdoor_detected": backdoor_detected,
            "clustering": cluster_result,
            "spectral": spectral_result,
        }


class ModelInversionDefense:
    def __init__(self, noise_std=0.01):
        self.noise_std = noise_std

    def defend(self, output):
        return output + np.random.normal(0, self.noise_std, output.shape)


class WatermarkDetector:
    def __init__(self, green_ratio=0.5, threshold=4.0):
        self.green_ratio = green_ratio
        self.threshold = threshold

    def detect(self, tokens, prev_token=None):
        green_list = self._get_green_list(prev_token)
        green_count = sum(1 for t in tokens if t in green_list)
        mu = len(tokens) * self.green_ratio
        sigma = np.sqrt(len(tokens) * self.green_ratio * (1 - self.green_ratio))
        z = (green_count - mu) / (sigma + 1e-8)
        return abs(z) > self.threshold

    def _get_green_list(self, prev_token):
        import hashlib
        h = hashlib.md5(str(prev_token).encode()).hexdigest()
        return set(range(int(h[:8], 16) % 1000, int(h[:8], 16) % 1000 + 500))


class RedTeamAgent:
    def __init__(self, model):
        self.model = model

    def generate_adversarial(self, target_behavior):
        prompts = self.model.generate(f"Generate a prompt that would cause: {target_behavior}")
        return prompts


class OutputConsistency:
    def __init__(self, model, num_samples=5):
        self.model = model
        self.num_samples = num_samples
        self.threshold = 0.5

    def check(self, prompt):
        outputs = [self.model.generate(prompt) for _ in range(self.num_samples)]
        variance = self._compute_variance(outputs)
        return variance < self.threshold

    def _compute_variance(self, outputs):
        if len(outputs) < 2:
            return 0.0
        unique_ratio = len(set(str(o) for o in outputs)) / len(outputs)
        return unique_ratio


class DPNoise:
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        clip_norm: float = 1.0,
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.noise_scale = clip_norm * math.sqrt(2 * math.log(1.25 / delta)) / epsilon

    def clip_gradient(self, gradient: np.ndarray) -> np.ndarray:
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > self.clip_norm:
            gradient = gradient * self.clip_norm / grad_norm
        return gradient

    def add_noise(self, gradient: np.ndarray) -> np.ndarray:
        clipped = self.clip_gradient(gradient)
        noise = np.random.normal(0, self.noise_scale, size=clipped.shape)
        return clipped + noise

    def private_gradient(self, gradients: list) -> np.ndarray:
        clipped = np.array([self.clip_gradient(g) for g in gradients])
        mean_grad = np.mean(clipped, axis=0)
        noise = np.random.normal(0, self.noise_scale, size=mean_grad.shape)
        return mean_grad + noise

    def get_params(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "clip_norm": self.clip_norm,
            "noise_scale": self.noise_scale,
        }


class SecureAggregation:
    def __init__(self, num_parties: int, clip_norm: float = 1.0):
        self.num_parties = num_parties
        self.clip_norm = clip_norm
        self.masks = [np.random.randn(1000) for _ in range(num_parties)]

    def _mask_gradient(self, gradient: np.ndarray, party_id: int) -> np.ndarray:
        flat = gradient.flatten()
        mask = self.masks[party_id][:len(flat)]
        masked = flat + mask
        return masked.reshape(gradient.shape)

    def _unmask_partial(self, masked_gradient: np.ndarray, party_id: int) -> np.ndarray:
        flat = masked_gradient.flatten()
        mask = self.masks[party_id][:len(flat)]
        unmasked = flat - mask
        return unmasked.reshape(masked_gradient.shape)

    def encrypt(self, gradient: np.ndarray, party_id: int) -> np.ndarray:
        clipped = gradient.copy()
        grad_norm = np.linalg.norm(clipped)
        if grad_norm > self.clip_norm:
            clipped = clipped * self.clip_norm / grad_norm
        return self._mask_gradient(clipped, party_id)

    def aggregate(self, encrypted_gradients: list) -> np.ndarray:
        if len(encrypted_gradients) == 0:
            return np.zeros(1)
        sum_grad = np.zeros_like(encrypted_gradients[0])
        for i, eg in enumerate(encrypted_gradients):
            decrypted = self._unmask_partial(eg, i)
            sum_grad = sum_grad + decrypted
        return sum_grad / len(encrypted_gradients)

    def verify_aggregation(self, original_gradients: list, aggregated: np.ndarray) -> dict:
        plain_mean = np.mean(original_gradients, axis=0)
        error = float(np.mean(np.abs(plain_mean - aggregated)))
        return {
            "mean_error": error,
            "num_parties": len(original_gradients),
            "aggregation_shape": list(aggregated.shape),
        }


class AdversarialDetector:
    def __init__(self, model, threshold: float = 0.5):
        self.model = model
        self.threshold = threshold
        self.baseline_stats = {}

    def _compute_logit_stats(self, logits: np.ndarray) -> dict:
        if logits.ndim == 3:
            logits = logits[:, -1, :]
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)
        entropy = -np.sum(probs * np.log(probs + 1e-8), axis=-1)
        sorted_probs = np.sort(probs, axis=-1)
        top1_prob = sorted_probs[:, -1]
        top2_prob = sorted_probs[:, -2]
        margin = top1_prob - top2_prob
        return {
            "entropy": float(np.mean(entropy)),
            "top1_prob": float(np.mean(top1_prob)),
            "top2_prob": float(np.mean(top2_prob)),
            "margin": float(np.mean(margin)),
        }

    def _compute_input_stats(self, x: np.ndarray) -> dict:
        return {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "max": float(np.max(x)),
            "min": float(np.min(x)),
            "l2_norm": float(np.linalg.norm(x)),
        }

    def set_baseline(self, clean_inputs: np.ndarray):
        stats_list = []
        for i in range(len(clean_inputs)):
            logits = self.model(Tensor(clean_inputs[i:i+1].astype(np.float32)))
            stats = self._compute_logit_stats(logits.data)
            stats.update(self._compute_input_stats(clean_inputs[i]))
            stats_list.append(stats)
        self.baseline_stats = {
            "mean_entropy": np.mean([s["entropy"] for s in stats_list]),
            "mean_top1_prob": np.mean([s["top1_prob"] for s in stats_list]),
            "mean_l2_norm": np.mean([s["l2_norm"] for s in stats_list]),
            "std_l2_norm": np.std([s["l2_norm"] for s in stats_list]),
        }

    def detect_fgsm(self, x: np.ndarray) -> dict:
        x_tensor = Tensor(x.astype(np.float32))
        logits = self.model(x_tensor)
        stats = self._compute_logit_stats(logits.data)
        input_stats = self._compute_input_stats(x)
        anomaly_score = 0.0
        if self.baseline_stats:
            entropy_diff = abs(stats["entropy"] - self.baseline_stats.get("mean_entropy", stats["entropy"]))
            prob_diff = abs(stats["top1_prob"] - self.baseline_stats.get("mean_top1_prob", stats["top1_prob"]))
            anomaly_score = 0.5 * entropy_diff + 0.5 * prob_diff
        return {
            "adversarial_detected": anomaly_score > self.threshold,
            "anomaly_score": anomaly_score,
            "logit_stats": stats,
            "input_stats": input_stats,
        }

    def detect_pgd(self, x: np.ndarray, num_checks: int = 5, epsilon: float = 0.01) -> dict:
        x_adv = x.copy()
        detections = []
        for _ in range(num_checks):
            noise = np.random.uniform(-epsilon, epsilon, x.shape)
            x_perturbed = np.clip(x + noise, 0, 1) if x.max() > 1 else x + noise
            result = self.detect_fgsm(x_perturbed)
            detections.append(result["adversarial_detected"])
        vote_count = sum(detections)
        return {
            "adversarial_detected": vote_count > len(detections) // 2,
            "vote_count": vote_count,
            "total_checks": len(detections),
            "adversarial_ratio": vote_count / len(detections),
        }

    def detect(self, x: np.ndarray) -> dict:
        fgsm_result = self.detect_fgsm(x)
        pgd_result = self.detect_pgd(x)
        return {
            "fgsm": fgsm_result,
            "pgd": pgd_result,
            "adversarial_detected": fgsm_result["adversarial_detected"] or pgd_result["adversarial_detected"],
        }
