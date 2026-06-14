"""Knowledge distillation. Pure Python + NumPy."""
import math
from typing import List, Optional, Tuple

import numpy as np

from qythera.tensor import Tensor
from qythera.nn import Module, Linear


# ---------------------------------------------------------------------------
# KDLoss – KL divergence soft targets + cross-entropy hard targets
# ---------------------------------------------------------------------------

class KDLoss(Module):
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, student_logits: Tensor, teacher_logits: Tensor,
                hard_targets: Optional[Tensor] = None) -> Tuple[Tensor, dict]:
        T = self.temperature
        soft_teacher = self._softmax(teacher_logits.data / T)
        soft_student = self._softmax(student_logits.data / T)
        log_soft_student = self._log_softmax(student_logits.data / T)
        kl = np.sum(soft_teacher * (np.log(soft_teacher + 1e-12) - log_soft_student))
        kl = kl * (T * T) / soft_teacher.shape[0]
        if hard_targets is not None:
            ce = self._cross_entropy(student_logits.data, hard_targets.data)
            loss_val = self.alpha * kl + (1.0 - self.alpha) * ce
        else:
            loss_val = kl
        loss = Tensor(np.array(loss_val, dtype=np.float32))
        return loss, {"kd_loss": float(kl), "temperature": T}

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x_max = np.max(x, axis=-1, keepdims=True)
        e = np.exp(x - x_max)
        return e / np.sum(e, axis=-1, keepdims=True)

    @staticmethod
    def _log_softmax(x: np.ndarray) -> np.ndarray:
        x_max = np.max(x, axis=-1, keepdims=True)
        log_e = x - x_max - np.log(np.sum(np.exp(x - x_max), axis=-1, keepdims=True))
        return log_e

    @staticmethod
    def _cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
        log_sm = KDLoss._log_softmax(logits)
        n = targets.shape[0]
        idx = np.arange(n)
        return -np.mean(log_sm[idx, targets])


# ---------------------------------------------------------------------------
# FeatureDistill – MSE between hidden states
# ---------------------------------------------------------------------------

class FeatureDistill(Module):
    def __init__(self, layers: Optional[List[int]] = None, projection_dim: Optional[int] = None):
        super().__init__()
        self.layers = layers
        self.projection_dim = projection_dim
        self._student_projs: List[Linear] = []
        self._teacher_projs: List[Linear] = []

    def _ensure_projs(self, student_dim: int, teacher_dim: int):
        if self.projection_dim is None:
            return
        if not self._student_projs:
            self._student_projs.append(Linear(student_dim, self.projection_dim))
            self._teacher_projs.append(Linear(teacher_dim, self.projection_dim))

    def forward(self, student_hiddens: List[Tensor], teacher_hiddens: List[Tensor]) -> Tuple[Tensor, dict]:
        if self.layers is not None:
            s_layers = [student_hiddens[i] for i in self.layers if i < len(student_hiddens)]
            t_layers = [teacher_hiddens[i] for i in self.layers if i < len(teacher_hiddens)]
        else:
            s_layers = student_hiddens
            t_layers = teacher_hiddens
            min_len = min(len(s_layers), len(t_layers))
            s_layers = s_layers[:min_len]
            t_layers = t_layers[:min_len]

        total_mse = 0.0
        count = 0
        for s, t in zip(s_layers, t_layers):
            sd, td = s.data, t.data
            min_shape = tuple(min(a, b) for a, b in zip(sd.shape, td.shape))
            sd_s = sd[tuple(slice(0, d) for d in min_shape)]
            td_s = td[tuple(slice(0, d) for d in min_shape)]
            total_mse += float(np.mean((sd_s - td_s) ** 2))
            count += 1
        loss_val = total_mse / max(count, 1)
        return Tensor(np.array(loss_val, dtype=np.float32)), {"feature_mse": loss_val}


# ---------------------------------------------------------------------------
# AttentionDistill – MSE between attention matrices
# ---------------------------------------------------------------------------

class AttentionDistill(Module):
    def __init__(self, layers: Optional[List[int]] = None):
        super().__init__()
        self.layers = layers

    def forward(self, student_attns: List[Tensor], teacher_attns: List[Tensor]) -> Tuple[Tensor, dict]:
        if self.layers is not None:
            s_attns = [student_attns[i] for i in self.layers if i < len(student_attns)]
            t_attns = [teacher_attns[i] for i in self.layers if i < len(teacher_attns)]
        else:
            min_len = min(len(student_attns), len(teacher_attns))
            s_attns = student_attns[:min_len]
            t_attns = teacher_attns[:min_len]

        total_mse = 0.0
        count = 0
        for s, t in zip(s_attns, t_attns):
            sd, td = s.data, t.data
            min_shape = tuple(min(a, b) for a, b in zip(sd.shape, td.shape))
            sd_s = sd[tuple(slice(0, d) for d in min_shape)]
            td_s = td[tuple(slice(0, d) for d in min_shape)]
            total_mse += float(np.mean((sd_s - td_s) ** 2))
            count += 1
        loss_val = total_mse / max(count, 1)
        return Tensor(np.array(loss_val, dtype=np.float32)), {"attn_mse": loss_val}


# ---------------------------------------------------------------------------
# DistillationTrainer – wraps teacher + student
# ---------------------------------------------------------------------------

class DistillationTrainer:
    def __init__(self, teacher: Module, student: Module, kd_loss: KDLoss,
                 feature_distill: Optional[FeatureDistill] = None,
                 attention_distill: Optional[AttentionDistill] = None,
                 feature_weight: float = 1.0, attention_weight: float = 1.0):
        self.teacher = teacher
        self.student = student
        self.kd_loss = kd_loss
        self.feature_distill = feature_distill
        self.attention_distill = attention_distill
        self.feature_weight = feature_weight
        self.attention_weight = attention_weight
        for p in self.teacher.parameters():
            p.requires_grad = False

    def train_step(self, inputs: Tensor, hard_targets: Optional[Tensor] = None,
                   return_hiddens: bool = False) -> Tuple[Tensor, dict]:
        with _no_grad_context():
            teacher_logits, teacher_hiddens, teacher_attns = self.teacher(inputs, return_hiddens=True)
        student_logits, student_hiddens, student_attns = self.student(inputs, return_hiddens=True)
        kd_loss, kd_info = self.kd_loss(student_logits, teacher_logits, hard_targets)
        total_loss = kd_loss
        info = {**kd_info}
        if self.feature_distill is not None and student_hiddens and teacher_hiddens:
            f_loss, f_info = self.feature_distill(student_hiddens, teacher_hiddens)
            total_loss = Tensor(total_loss.data + self.feature_weight * f_loss.data)
            info.update(f_info)
        if self.attention_distill is not None and student_attns and teacher_attns:
            a_loss, a_info = self.attention_distill(student_attns, teacher_attns)
            total_loss = Tensor(total_loss.data + self.attention_weight * a_loss.data)
            info.update(a_info)
        if return_hiddens:
            info["student_logits"] = student_logits
        return total_loss, info


class _no_grad_context:
    def __enter__(self):
        from qythera.tensor import no_grad
        self.ctx = no_grad()
        self.ctx.__enter__()
        return self

    def __exit__(self, *a):
        self.ctx.__exit__(*a)


# ---------------------------------------------------------------------------
# MedusaHeads – parallel prediction heads for +1..+K
# ---------------------------------------------------------------------------

class MedusaHeads(Module):
    def __init__(self, hidden_dim: int, vocab_size: int, num_heads: int = 3):
        super().__init__()
        self.heads = []
        for i in range(num_heads):
            head = Linear(hidden_dim, vocab_size)
            self.heads.append(head)
            self._modules[f"head_{i}"] = head

    def forward(self, hidden_state: Tensor) -> List[Tensor]:
        return [head(hidden_state) for head in self.heads]

    def get_predictions(self, hidden_state: Tensor) -> List[Tensor]:
        logits_list = self.forward(hidden_state)
        return [Tensor(np.argmax(lg.data, axis=-1)) for lg in logits_list]


# ---------------------------------------------------------------------------
# MultiTokenPrediction – auxiliary heads during training
# ---------------------------------------------------------------------------

class MultiTokenPrediction(Module):
    def __init__(self, hidden_dim: int, vocab_size: int, num_future: int = 3):
        super().__init__()
        self.heads = []
        for i in range(num_future):
            head = Linear(hidden_dim, vocab_size)
            self.heads.append(head)
            self._modules[f"mtp_head_{i}"] = head

    def forward(self, hidden_state: Tensor) -> List[Tensor]:
        return [head(hidden_state) for head in self.heads]

    def compute_loss(self, hidden_state: Tensor, targets: List[Tensor],
                     base_loss_fn) -> Tuple[Tensor, dict]:
        logits_list = self.forward(hidden_state)
        total_loss = 0.0
        details = {}
        for i, (lg, tgt) in enumerate(zip(logits_list, targets)):
            if i < len(targets):
                loss_val = base_loss_fn(lg.data, tgt.data)
                total_loss += loss_val
                details[f"mtp_loss_{i}"] = float(loss_val)
        n_heads = max(len(targets), 1)
        total_loss = total_loss / n_heads
        return Tensor(np.array(total_loss, dtype=np.float32)), details


# ---------------------------------------------------------------------------
# SpeculativeDecoder – draft generates K tokens, target verifies
# ---------------------------------------------------------------------------

class SpeculativeDecoder:
    def __init__(self, draft_model: Module, target_model: Module, k: int = 5,
                 temperature: float = 1.0):
        self.draft = draft_model
        self.target = target_model
        self.k = k
        self.temperature = temperature

    def _sample(self, logits: np.ndarray, temperature: float = 1.0) -> int:
        scaled = logits / max(temperature, 1e-8)
        scaled = scaled - np.max(scaled)
        probs = np.exp(scaled) / (np.sum(np.exp(scaled)) + 1e-12)
        return int(np.random.choice(len(probs), p=probs))

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x_max = np.max(x, axis=-1, keepdims=True)
        e = np.exp(x - x_max)
        return e / (np.sum(e, axis=-1, keepdims=True) + 1e-12)

    def generate(self, prompt: Tensor, max_new_tokens: int = 100) -> Tensor:
        generated = list(prompt.data.flatten())
        while len(generated) - len(prompt.data.flatten()) < max_new_tokens:
            draft_tokens = []
            draft_probs = []
            state = np.array([generated], dtype=np.int64)
            for _ in range(self.k):
                inp = Tensor(state)
                with _no_grad_context():
                    draft_logits = self.draft(inp)
                probs = self._softmax(draft_logits.data[0, -1])
                tok = self._sample(probs, self.temperature)
                draft_tokens.append(tok)
                draft_probs.append(probs[tok])
                state = np.concatenate([state, [[tok]]], axis=1)

            inp = Tensor(state)
            with _no_grad_context():
                target_logits = self.target(inp)
            target_probs = self._softmax(target_logits.data[0, -self.k - 1:])

            accepted = 0
            for i in range(self.k):
                t_prob = target_probs[i, draft_tokens[i]]
                d_prob = draft_probs[i]
                accept_ratio = min(1.0, t_prob / (d_prob + 1e-12))
                if np.random.random() < accept_ratio:
                    generated.append(draft_tokens[i])
                    accepted += 1
                else:
                    corrected = self._sample(target_probs[i], self.temperature)
                    generated.append(corrected)
                    break
            else:
                if len(generated) - len(prompt.data.flatten()) < max_new_tokens:
                    inp = Tensor(np.array([generated], dtype=np.int64))
                    with _no_grad_context():
                        final_logits = self.target(inp)
                    next_tok = self._sample(
                        self._softmax(final_logits.data[0, -1]), self.temperature
                    )
                    generated.append(next_tok)

        total_new = max_new_tokens
        return Tensor(np.array([generated[:len(prompt.data.flatten()) + total_new]], dtype=np.int64))

    def verify(self, draft_tokens: List[int], target_logits: Tensor) -> Tuple[List[int], int]:
        probs = self._softmax(target_logits.data)
        accepted = []
        for i, tok in enumerate(draft_tokens):
            if i >= probs.shape[0]:
                break
            p_target = probs[i, tok]
            p_draft = 1.0 / probs.shape[-1]
            if np.random.random() < min(1.0, p_target / (p_draft + 1e-12)):
                accepted.append(tok)
            else:
                corrected = self._sample(probs[i], self.temperature)
                accepted.append(corrected)
                return accepted, i + 1
        if len(draft_tokens) < probs.shape[0]:
            extra = self._sample(probs[len(draft_tokens)], self.temperature)
            accepted.append(extra)
        return accepted, len(draft_tokens)
