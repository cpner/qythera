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


class SpeculativeDecoder:
    def __init__(self, draft_model, target_model, draft_tokens: int = 5,
                 temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0):
        self.draft_model = draft_model
        self.target_model = target_model
        self.draft_tokens = draft_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def _sample_from_logits(self, logits: np.ndarray) -> int:
        if self.temperature > 0:
            logits = logits / max(self.temperature, 0.01)
        if self.top_k > 0:
            threshold = np.sort(logits)[-min(self.top_k, len(logits))]
            logits = logits.copy()
            logits[logits < threshold] = -np.inf
        if self.top_p < 1.0:
            sorted_idx = np.argsort(logits)[::-1]
            sorted_logits = logits[sorted_idx].copy()
            cum_probs = np.cumsum(softmax(sorted_logits))
            mask = cum_probs > self.top_p
            mask[1:] = mask[:-1]
            mask[0] = False
            sorted_logits[mask] = -np.inf
            logits[sorted_idx] = sorted_logits
        probs = softmax(logits)
        return int(np.random.choice(len(probs), p=probs))

    def generate(self, prompt_ids: List[int], max_new_tokens: int = 128) -> List[int]:
        from qythera.tensor import Tensor
        import numpy as np
        ids = list(prompt_ids)
        generated = []

        while len(generated) < max_new_tokens:
            remaining = max_new_tokens - len(generated)
            k = min(self.draft_tokens, remaining)

            draft_ids = []
            draft_probs = []
            draft_cache = self.draft_model.init_kv_cache()

            inp = Tensor(np.array([ids], dtype=np.int32))
            draft_logits = self.draft_model.forward(inp, kv_cache=draft_cache, position=0)
            pos = draft_cache.get_seq_len()

            for _ in range(k):
                last = draft_logits.data[0, -1]
                token = self._sample_from_logits(last)
                draft_ids.append(token)
                probs = softmax(last / max(self.temperature, 0.01))
                draft_probs.append(probs)

                inp = Tensor(np.array([[token]], dtype=np.int32))
                draft_logits = self.draft_model.forward(inp, kv_cache=draft_cache, position=pos)
                pos = draft_cache.get_seq_len()

            draft_cache.reset()

            all_ids = ids + draft_ids
            inp = Tensor(np.array([all_ids], dtype=np.int32))
            target_cache = self.target_model.init_kv_cache()
            target_logits = self.target_model.forward(inp, kv_cache=target_cache, position=0)
            target_cache.reset()

            accepted = 0
            for i in range(k):
                t = target_logits.data[0, len(ids) + i - 1]
                t_prob = softmax(t / max(self.temperature, 0.01))
                p_draft = draft_probs[i][draft_ids[i]]
                p_target = t_prob[draft_ids[i]]
                ratio = min(1.0, p_target / max(p_draft, 1e-10))

                if np.random.random() < ratio:
                    generated.append(draft_ids[i])
                    ids.append(draft_ids[i])
                else:
                    sampled = self._sample_from_logits(t)
                    generated.append(sampled)
                    ids.append(sampled)
                    break
            else:
                last_t = target_logits.data[0, -1]
                bonus = self._sample_from_logits(last_t)
                generated.append(bonus)
                ids.append(bonus)

        return ids[:len(prompt_ids) + max_new_tokens]


class MedusaHeads:
    def __init__(self, vocab_size: int, hidden_dim: int, num_heads: int = 4,
                 top_k: int = 10):
        from qythera.nn import Linear, Module, ModuleList
        self.num_heads = num_heads
        self.top_k = top_k
        self.vocab_size = vocab_size
        self.heads = ModuleList([Linear(hidden_dim, vocab_size, bias=False)
                                 for _ in range(num_heads)])
        self._head_weights = [np.ones(1, dtype=np.float32) for _ in range(num_heads)]

    def predict(self, hidden_states: np.ndarray, temperature: float = 1.0,
                top_k: Optional[int] = None) -> List[List[int]]:
        top_k = top_k or self.top_k
        candidates = []
        from qythera.tensor import Tensor
        for i in range(self.num_heads):
            inp = Tensor(hidden_states)
            logits = self.heads[i](inp).data[0, -1]
            if temperature > 0:
                logits = logits / max(temperature, 0.01)
            topk_idx = np.argsort(logits)[-top_k:][::-1]
            candidates.append(topk_idx.tolist())
        return candidates

    def tree_attention(self, candidates: List[List[int]],
                       hidden_states: np.ndarray) -> np.ndarray:
        num_heads = len(candidates)
        max_depth = min(num_heads, 3)
        paths = [[]]
        for depth in range(max_depth):
            new_paths = []
            for path in paths:
                for token in candidates[depth][:self.top_k]:
                    new_paths.append(path + [token])
            paths = new_paths
            if len(paths) > 64:
                paths = paths[:64]

        path_scores = np.zeros(len(paths), dtype=np.float32)
        from qythera.tensor import Tensor
        for i in range(num_heads):
            inp = Tensor(hidden_states)
            logits = self.heads[i](inp).data[0, -1]
            probs = softmax(logits / max(1.0, 0.01))
            for j, path in enumerate(paths):
                if i < len(path):
                    path_scores[j] += np.log(probs[path[i]] + 1e-10)

        sorted_idx = np.argsort(path_scores)[::-1]
        return np.array([paths[i] for i in sorted_idx[:self.top_k]])

    def reset(self):
        pass


class MCTSNode:
    def __init__(self, token_id: int, parent: Optional['MCTSNode'] = None):
        self.token_id = token_id
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.value = 0.0
        self.prob = 0.0
        self.is_expanded = False

    def ucb1(self, exploration: float = 1.414) -> float:
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + exploration * np.sqrt(np.log(self.parent.visits) / self.visits)


class MCTSGenerator:
    def __init__(self, model, reward_model, num_simulations: int = 10,
                 exploration: float = 1.414, top_k: int = 5):
        self.model = model
        self.reward_model = reward_model
        self.num_simulations = num_simulations
        self.exploration = exploration
        self.top_k = top_k

    def generate(self, prompt_ids: List[int], max_tokens: int = 50) -> List[int]:
        root = MCTSNode(token_id=-1)
        generated: List[int] = list(prompt_ids)

        for step in range(max_tokens):
            for _ in range(self.num_simulations):
                node = self._select(root)
                child = self._expand(node, generated)
                reward = self._simulate(generated + [child.token_id])
                self._backprop(child, reward)

            best_child = max(root.children, key=lambda c: c.visits)
            generated.append(best_child.token_id)
            root = best_child
            root.parent = None

        return generated

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.is_expanded and node.children:
            node = max(node.children, key=lambda c: c.ucb1(self.exploration))
        return node

    def _expand(self, node: MCTSNode, context: List[int]) -> MCTSNode:
        if node.is_expanded:
            return node
        node.is_expanded = True

        from qythera.tensor import Tensor
        inp = Tensor(np.array([context], dtype=np.int32))
        logits = self.model.forward(inp)
        probs = softmax(logits.data[0, -1])
        top_indices = np.argsort(probs)[-self.top_k:]

        for idx in top_indices:
            child = MCTSNode(token_id=int(idx), parent=node)
            child.prob = float(probs[idx])
            node.children.append(child)

        if not node.children:
            child = MCTSNode(token_id=int(np.argmax(probs)), parent=node)
            child.prob = float(np.max(probs))
            node.children.append(child)

        return node.children[0]

    def _simulate(self, context: List[int]) -> float:
        return self.reward_model.score(context)

    def _backprop(self, node: MCTSNode, reward: float):
        current = node
        while current:
            current.visits += 1
            current.value += reward
            current = current.parent


class DiverseBeamSearch:
    def __init__(self, model, beam_width: int = 5, diversity_penalty: float = 0.5,
                 length_penalty: float = 0.6):
        self.model = model
        self.beam_width = beam_width
        self.diversity_penalty = diversity_penalty
        self.length_penalty = length_penalty

    def generate(self, prompt_ids: List[int], max_tokens: int = 50) -> List[List[int]]:
        beams: List[Tuple[List[int], float]] = [(list(prompt_ids), 0.0)]
        completed: List[Tuple[List[int], float]] = []

        for _ in range(max_tokens):
            all_candidates: List[Tuple[List[int], float]] = []
            seen_tokens_per_beam: List[set] = [set() for _ in beams]

            for beam_idx, (seq, score) in enumerate(beams):
                if len(seq) > 0 and seq[-1] == 0:
                    completed.append((seq, score))
                    continue

                from qythera.tensor import Tensor
                inp = Tensor(np.array([seq], dtype=np.int32))
                logits = self.model.forward(inp)
                probs = softmax(logits.data[0, -1])

                for i in range(self.beam_width):
                    token_idx = int(np.argmax(probs))
                    token_prob = float(probs[token_idx])

                    diversity_penalty = 0.0
                    for other_idx, other_seen in enumerate(seen_tokens_per_beam):
                        if other_idx != beam_idx and token_idx in other_seen:
                            diversity_penalty += self.diversity_penalty

                    new_score = score + np.log(token_prob + 1e-10) - diversity_penalty
                    all_candidates.append((seq + [token_idx], new_score))
                    probs = probs.copy()
                    probs[token_idx] = -np.inf

            if not all_candidates:
                break

            all_candidates.extend(completed)
            all_candidates.sort(key=lambda x: x[1], reverse=True)

            beams = []
            seen_tokens_per_beam = []
            for seq, score in all_candidates[:self.beam_width]:
                if len(seq) > 0 and seq[-1] == 0:
                    completed.append((seq, score))
                else:
                    beams.append((seq, score))
                    seen_tokens_per_beam.append(set(seq[len(prompt_ids):]))

            if not beams:
                break

        results = completed if completed else beams
        normalized = []
        for seq, score in results:
            length = len(seq) - len(prompt_ids)
            norm_score = score / max(length ** self.length_penalty, 1e-10)
            normalized.append((seq, norm_score))
        normalized.sort(key=lambda x: x[1], reverse=True)
        return [seq for seq, _ in normalized[:self.beam_width]]


class ContrastiveSearch:
    def __init__(self, model, alpha: float = 0.6, top_k: int = 5):
        self.model = model
        self.alpha = alpha
        self.top_k = top_k

    def generate(self, prompt_ids: List[int], max_tokens: int = 50) -> List[int]:
        generated = list(prompt_ids)

        for _ in range(max_tokens):
            from qythera.tensor import Tensor

            inp = Tensor(np.array([generated], dtype=np.int32))
            logits = self.model.forward(inp)
            next_token_logits = logits.data[0, -1]

            top_k_idx = np.argsort(next_token_logits)[-self.top_k:]

            scores = np.zeros(self.top_k)
            for i, idx in enumerate(top_k_idx):
                token_logit = next_token_logits[idx]
                other_logits = np.delete(next_token_logits, idx)
                max_other_logit = np.max(other_logits)
                degeneration_penalty = max(0, token_logit - max_other_logit)
                scores[i] = (1 - self.alpha) * token_logit - self.alpha * degeneration_penalty

            best_idx = top_k_idx[np.argmax(scores)]
            generated.append(int(best_idx))

        return generated


class ContinuousBatcher:
    def __init__(self, max_batch_size: int = 32, max_seq_len: int = 2048,
                 pad_token_id: int = 0):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.slots: Dict[int, Dict[str, Any]] = {}
        self._slot_counter = 0
        self._kv_caches: Dict[int, Any] = {}
        self._positions: Dict[int, int] = {}
        self._completed: Dict[int, List[int]] = {}
        self._prompt_lengths: Dict[int, int] = {}

    def add_request(self, token_ids: List[int]) -> int:
        slot_id = self._slot_counter
        self._slot_counter += 1
        self.slots[slot_id] = {
            "input_ids": token_ids,
            "status": "prefill",
            "max_new_tokens": 1024,
            "generated": [],
        }
        self._positions[slot_id] = 0
        self._prompt_lengths[slot_id] = len(token_ids)
        return slot_id

    def remove_request(self, slot_id: int):
        self.slots.pop(slot_id, None)
        self._kv_caches.pop(slot_id, None)
        self._positions.pop(slot_id, None)
        self._completed.pop(slot_id, None)
        self._prompt_lengths.pop(slot_id, None)

    def is_complete(self, slot_id: int) -> bool:
        slot = self.slots.get(slot_id)
        if slot is None:
            return True
        return len(slot["generated"]) >= slot.get("max_new_tokens", 1024)

    def get_active_slots(self) -> List[int]:
        return [sid for sid in self.slots if not self.is_complete(sid)]

    def prepare_batch(self) -> Optional[Tuple[np.ndarray, List[int]]]:
        active = self.get_active_slots()
        if not active:
            return None
        active = active[:self.max_batch_size]

        batch_ids = []
        for sid in active:
            slot = self.slots[sid]
            if slot["status"] == "prefill":
                batch_ids.append(slot["input_ids"])
            else:
                last = slot["generated"][-1] if slot["generated"] else slot["input_ids"][-1]
                batch_ids.append([last])

        max_len = max(len(ids) for ids in batch_ids)
        padded = np.full((len(batch_ids), max_len), self.pad_token_id, dtype=np.int32)
        attention_mask = np.zeros((len(batch_ids), max_len), dtype=np.float32)

        for i, ids in enumerate(batch_ids):
            padded[i, max_len - len(ids):] = ids
            attention_mask[i, max_len - len(ids):] = 1.0

        return padded, attention_mask

    def update_with_output(self, slot_ids: List[int], token_ids: List[int]):
        for sid, token_id in zip(slot_ids, token_ids):
            if sid not in self.slots:
                continue
            slot = self.slots[sid]
            if slot["status"] == "prefill":
                slot["status"] = "decode"
                self._positions[sid] = self._prompt_lengths[sid]
            slot["generated"].append(int(token_id))
            self._positions[sid] += 1

            if self.is_complete(sid):
                self._completed[sid] = list(slot["generated"])

    def get_completed(self) -> Dict[int, List[int]]:
        result = dict(self._completed)
        self._completed.clear()
        return result

    def get_batch_stats(self) -> Dict[str, Any]:
        active = self.get_active_slots()
        return {
            "total_slots": len(self.slots),
            "active_slots": len(active),
            "completed_slots": len(self._completed),
            "utilization": len(active) / max(self.max_batch_size, 1),
        }
