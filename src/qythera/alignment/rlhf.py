import math
import copy
import numpy as np
from typing import Optional, List, Tuple, Any

from qythera.tensor import Tensor, no_grad
from qythera.nn import Module, Linear
from qythera.model import Transformer, TransformerConfig
from qythera.optim import Adam


def _log_softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    m = logits.max(axis=axis, keepdims=True)
    e = np.exp(logits - m)
    return logits - m - np.log(e.sum(axis=axis, keepdims=True) + 1e-8)


def _gather_log_probs(log_probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    B, L = labels.shape
    return log_probs[np.arange(B)[:, None], np.arange(L)[None, :], labels].sum(axis=-1)


def _model_log_probs(model: Module, input_ids: np.ndarray) -> np.ndarray:
    with no_grad():
        logits = model(Tensor(input_ids.astype(np.int32)))
    return _log_softmax_np(logits.data, axis=-1)


def _model_logits(model: Module, input_ids: np.ndarray) -> np.ndarray:
    with no_grad():
        logits = model(Tensor(input_ids.astype(np.int32)))
    return logits.data


class RewardModel(Module):
    def __init__(self, backbone: Transformer):
        super().__init__()
        self.backbone = backbone
        self.head = Linear(backbone.config.vocab_size, 1, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        hidden = self.backbone(input_ids)
        last_hidden = hidden[:, -1, :]
        reward = self.head(last_hidden)
        return reward.squeeze(-1)

    def forward_pair(self, chosen_ids: Tensor, rejected_ids: Tensor) -> Tuple[float, np.ndarray, np.ndarray]:
        with no_grad():
            r_chosen = self.forward(chosen_ids)
            r_rejected = self.forward(rejected_ids)
        r_c = r_chosen.data
        r_r = r_rejected.data
        loss = -np.log(1.0 / (1.0 + np.exp(-r_c)) + 1e-8) - np.log(1.0 - 1.0 / (1.0 + np.exp(-r_r)) + 1e-8)
        return float(loss.mean()), r_c, r_r

    def score(self, input_ids: Tensor) -> np.ndarray:
        with no_grad():
            reward = self.forward(input_ids)
        return reward.data

    def bradley_terry_loss(self, chosen_rewards: np.ndarray, rejected_rewards: np.ndarray) -> float:
        diff = chosen_rewards - rejected_rewards
        loss = -np.log(1.0 / (1.0 + np.exp(-diff)) + 1e-8)
        return float(loss.mean())


class RolloutBuffer:
    def __init__(self):
        self.observations: List[Any] = []
        self.actions: List[Any] = []
        self.log_probs: List[Any] = []
        self.rewards: List[Any] = []
        self.values: List[Any] = []
        self.advantages: List[Any] = []
        self.returns: List[Any] = []
        self.dones: List[Any] = []

    def add(self, obs, action, log_prob, reward, value, done=False):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_value: float, gamma: float = 1.0, lam: float = 0.95):
        advantages = []
        returns = []
        last_gae = 0.0
        values = [float(v) for v in self.values]
        rewards = [float(r) for r in self.rewards]
        dones = [float(d) for d in self.dones]

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t] if t < len(rewards) - 1 else 1.0
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages.insert(0, last_gae)
            returns.insert(0, last_gae + values[t])

        self.advantages = advantages
        self.returns = returns

    def get_batches(self, batch_size: int):
        n = len(self.observations)
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            yield {
                "observations": [self.observations[i] for i in batch_idx],
                "actions": [self.actions[i] for i in batch_idx],
                "log_probs": [self.log_probs[i] for i in batch_idx],
                "advantages": [self.advantages[i] for i in batch_idx],
                "returns": [self.returns[i] for i in batch_idx],
            }

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.advantages.clear()
        self.returns.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.observations)


class PPOTrainer:
    def __init__(
        self,
        policy: Module,
        value_net: Module,
        lr: float = 3e-4,
        gamma: float = 1.0,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        vf_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
    ):
        self.policy = policy
        self.value_net = value_net
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.vf_coeff = vf_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.policy_optim = Adam(policy.parameters(), lr=lr)
        self.value_optim = Adam(value_net.parameters(), lr=lr)

    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool], last_value: float) -> Tuple[List[float], List[float]]:
        advantages = []
        returns = []
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t < len(rewards) - 1 else last_value
            next_non_terminal = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.lam * next_non_terminal * last_gae
            advantages.insert(0, last_gae)
            returns.insert(0, last_gae + values[t])
        return advantages, returns

    def clipped_ppo_loss(self, log_ratio: np.ndarray, advantages: np.ndarray) -> float:
        ratio = np.exp(log_ratio)
        clipped = np.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        loss1 = ratio * advantages
        loss2 = clipped * advantages
        return -float(np.minimum(loss1, loss2).mean())

    def value_loss(self, values: np.ndarray, returns: np.ndarray) -> float:
        return float(((values - returns) ** 2).mean())

    def entropy_bonus(self, logits: np.ndarray) -> float:
        probs = np.exp(_log_softmax_np(logits, axis=-1))
        entropy = -np.sum(probs * _log_softmax_np(logits, axis=-1), axis=-1)
        return float(entropy.mean())

    def update(self, rollout_buffer: RolloutBuffer) -> dict:
        self.policy.train()
        self.value_net.train()

        all_policy_losses = []
        all_value_losses = []
        all_entropies = []
        all_returns_list = []
        all_advantages_list = []

        for _ in range(self.n_epochs):
            for batch in rollout_buffer.get_batches(self.batch_size):
                obs_batch = batch["observations"]
                actions_batch = batch["actions"]
                old_log_probs_batch = batch["log_probs"]
                advantages_batch = batch["advantages"]
                returns_batch = batch["returns"]

                obs_np = np.array([o if isinstance(o, np.ndarray) else np.array(o) for o in obs_batch], dtype=np.float32)
                actions_np = np.array(actions_batch, dtype=np.int32)
                old_log_probs_np = np.array(old_log_probs_batch, dtype=np.float32)
                advantages_np = np.array(advantages_batch, dtype=np.float32)
                returns_np = np.array(returns_batch, dtype=np.float32)

                advantages_np = (advantages_np - advantages_np.mean()) / (advantages_np.std() + 1e-8)

                with no_grad():
                    logits = self.policy(Tensor(obs_np.astype(np.int32)))
                    values = self.value_net(Tensor(obs_np.astype(np.int32))).squeeze(-1)

                log_probs_np = _log_softmax_np(logits.data, axis=-1)
                B = len(actions_np)
                token_log_probs_np = log_probs_np[np.arange(B)[:, None], np.arange(log_probs_np.shape[1])[None, :], actions_np].sum(axis=-1)
                log_ratio = token_log_probs_np - old_log_probs_np

                p_loss = self.clipped_ppo_loss(log_ratio, advantages_np)
                v_loss = self.value_loss(values.data, returns_np)
                entropy = self.entropy_bonus(logits.data)

                total_loss = p_loss + self.vf_coeff * v_loss - self.entropy_coeff * entropy

                for p in self.policy.parameters():
                    if p.grad is not None:
                        p.grad = None
                for p in self.value_net.parameters():
                    if p.grad is not None:
                        p.grad = None

                all_policy_losses.append(p_loss)
                all_value_losses.append(v_loss)
                all_entropies.append(entropy)
                all_returns_list.append(float(returns_np.mean()))
                all_advantages_list.append(float(advantages_np.mean()))

        return {
            "policy_loss": np.mean(all_policy_losses) if all_policy_losses else 0.0,
            "value_loss": np.mean(all_value_losses) if all_value_losses else 0.0,
            "entropy": np.mean(all_entropies) if all_entropies else 0.0,
            "mean_return": np.mean(all_returns_list) if all_returns_list else 0.0,
            "mean_advantage": np.mean(all_advantages_list) if all_advantages_list else 0.0,
        }


class DPOTrainer:
    def __init__(
        self,
        policy: Module,
        ref_model: Module,
        beta: float = 0.1,
        lr: float = 1e-6,
        max_grad_norm: float = 1.0,
    ):
        self.policy = policy
        self.ref_model = ref_model
        self.beta = beta
        self.max_grad_norm = max_grad_norm
        self.optimizer = Adam(policy.parameters(), lr=lr)
        self.ref_model.eval()

    def train_step(
        self,
        chosen_ids: Tensor,
        rejected_ids: Tensor,
    ) -> float:
        self.policy.train()

        chosen_np = chosen_ids.data.astype(np.int32) if isinstance(chosen_ids, Tensor) else np.array(chosen_ids, dtype=np.int32)
        rejected_np = rejected_ids.data.astype(np.int32) if isinstance(rejected_ids, Tensor) else np.array(rejected_ids, dtype=np.int32)

        policy_chosen_lp = _gather_log_probs(_model_log_probs(self.policy, chosen_np), chosen_np)
        policy_rejected_lp = _gather_log_probs(_model_log_probs(self.policy, rejected_np), rejected_np)

        ref_chosen_lp = _gather_log_probs(_model_log_probs(self.ref_model, chosen_np), chosen_np)
        ref_rejected_lp = _gather_log_probs(_model_log_probs(self.ref_model, rejected_np), rejected_np)

        chosen_log_ratios = policy_chosen_lp - ref_chosen_lp
        rejected_log_ratios = policy_rejected_lp - ref_rejected_lp

        logits = self.beta * (chosen_log_ratios - rejected_log_ratios)
        loss = -np.mean(np.log(1.0 / (1.0 + np.exp(-logits)) + 1e-8))

        self.policy.zero_grad()
        self._approximate_backward(self.policy, chosen_np, rejected_np,
                                   policy_chosen_lp, policy_rejected_lp,
                                   ref_chosen_lp, ref_rejected_lp, loss)
        self.optimizer.step()

        return float(loss)

    def _approximate_backward(self, policy, chosen_np, rejected_np,
                              policy_chosen_lp, policy_rejected_lp,
                              ref_chosen_lp, ref_rejected_lp, loss_val):
        eps = 1e-4
        for p in policy.parameters():
            if not p.requires_grad:
                continue
            grad = np.zeros_like(p.data)
            original = p.data.copy()
            flat = p.data.flatten()
            n_params = min(len(flat), max(10, len(flat) // 100))
            indices = np.random.choice(len(flat), n_params, replace=False)
            for idx in indices:
                flat[idx] += eps
                p.data = flat.reshape(p.data.shape)
                lp_c = _gather_log_probs(_model_log_probs(policy, chosen_np), chosen_np)
                lp_r = _gather_log_probs(_model_log_probs(policy, rejected_np), rejected_np)
                logits_p = self.beta * ((lp_c - ref_chosen_lp) - (lp_r - ref_rejected_lp))
                loss_p = -np.mean(np.log(1.0 / (1.0 + np.exp(-logits_p)) + 1e-8))
                flat_grad = np.zeros_like(flat)
                flat_grad[idx] = (loss_p - loss_val) / eps
                grad += flat_grad.reshape(grad.shape)
                flat[idx] = original.flatten()[idx]
            p.data = original
            p.grad = grad / n_params


class SimPOTrainer:
    def __init__(
        self,
        policy: Module,
        beta: float = 2.0,
        gamma: float = 0.5,
        lr: float = 1e-6,
        max_grad_norm: float = 1.0,
    ):
        self.policy = policy
        self.beta = beta
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.optimizer = Adam(policy.parameters(), lr=lr)

    def train_step(
        self,
        chosen_ids: Tensor,
        rejected_ids: Tensor,
    ) -> float:
        self.policy.train()

        chosen_np = chosen_ids.data.astype(np.int32) if isinstance(chosen_ids, Tensor) else np.array(chosen_ids, dtype=np.int32)
        rejected_np = rejected_ids.data.astype(np.int32) if isinstance(rejected_ids, Tensor) else np.array(rejected_ids, dtype=np.int32)

        policy_chosen_lp = _gather_log_probs(_model_log_probs(self.policy, chosen_np), chosen_np)
        policy_rejected_lp = _gather_log_probs(_model_log_probs(self.policy, rejected_np), rejected_np)

        seq_len_chosen = float(chosen_np.shape[1])
        seq_len_rejected = float(rejected_np.shape[1])

        rewards_chosen = policy_chosen_lp / seq_len_chosen
        rewards_rejected = policy_rejected_lp / seq_len_rejected

        logits = self.beta * (rewards_chosen - rewards_rejected) - self.gamma
        loss = -np.mean(np.log(1.0 / (1.0 + np.exp(-logits)) + 1e-8))

        self.policy.zero_grad()
        self._approximate_backward(self.policy, chosen_np, rejected_np,
                                   policy_chosen_lp, policy_rejected_lp, loss)
        self.optimizer.step()

        return float(loss)

    def _approximate_backward(self, policy, chosen_np, rejected_np,
                              policy_chosen_lp, policy_rejected_lp, loss_val):
        eps = 1e-4
        for p in policy.parameters():
            if not p.requires_grad:
                continue
            grad = np.zeros_like(p.data)
            original = p.data.copy()
            flat = p.data.flatten()
            n_params = min(len(flat), max(10, len(flat) // 100))
            indices = np.random.choice(len(flat), n_params, replace=False)
            seq_c = float(chosen_np.shape[1])
            seq_r = float(rejected_np.shape[1])
            for idx in indices:
                flat[idx] += eps
                p.data = flat.reshape(p.data.shape)
                lp_c = _gather_log_probs(_model_log_probs(policy, chosen_np), chosen_np)
                lp_r = _gather_log_probs(_model_log_probs(policy, rejected_np), rejected_np)
                logits_p = self.beta * ((lp_c / seq_c) - (lp_r / seq_r)) - self.gamma
                loss_p = -np.mean(np.log(1.0 / (1.0 + np.exp(-logits_p)) + 1e-8))
                flat_grad = np.zeros_like(flat)
                flat_grad[idx] = (loss_p - loss_val) / eps
                grad += flat_grad.reshape(grad.shape)
                flat[idx] = original.flatten()[idx]
            p.data = original
            p.grad = grad / n_params


class ORPOTrainer:
    def __init__(
        self,
        policy: Module,
        beta: float = 0.1,
        lr: float = 1e-6,
        max_grad_norm: float = 1.0,
    ):
        self.policy = policy
        self.beta = beta
        self.max_grad_norm = max_grad_norm
        self.optimizer = Adam(policy.parameters(), lr=lr)

    def train_step(
        self,
        chosen_ids: Tensor,
        rejected_ids: Tensor,
    ) -> float:
        self.policy.train()

        chosen_np = chosen_ids.data.astype(np.int32) if isinstance(chosen_ids, Tensor) else np.array(chosen_ids, dtype=np.int32)
        rejected_np = rejected_ids.data.astype(np.int32) if isinstance(rejected_ids, Tensor) else np.array(rejected_ids, dtype=np.int32)

        policy_chosen_lp = _gather_log_probs(_model_log_probs(self.policy, chosen_np), chosen_np)
        policy_rejected_lp = _gather_log_probs(_model_log_probs(self.policy, rejected_np), rejected_np)

        sft_loss = -np.mean(policy_chosen_lp)

        chosen_probs = np.exp(policy_chosen_lp)
        rejected_probs = np.exp(policy_rejected_lp)
        odds_chosen = chosen_probs / (1.0 - chosen_probs + 1e-8)
        odds_rejected = rejected_probs / (1.0 - rejected_probs + 1e-8)
        log_odds_ratio = np.log(odds_chosen / (odds_rejected + 1e-8) + 1e-8)
        or_loss = -np.mean(np.log(1.0 / (1.0 + np.exp(-log_odds_ratio)) + 1e-8))

        loss = sft_loss + self.beta * or_loss

        self.policy.zero_grad()
        self._approximate_backward(self.policy, chosen_np, rejected_np,
                                   policy_chosen_lp, policy_rejected_lp, loss)
        self.optimizer.step()

        return float(loss)

    def _approximate_backward(self, policy, chosen_np, rejected_np,
                              policy_chosen_lp, policy_rejected_lp, loss_val):
        eps = 1e-4
        for p in policy.parameters():
            if not p.requires_grad:
                continue
            grad = np.zeros_like(p.data)
            original = p.data.copy()
            flat = p.data.flatten()
            n_params = min(len(flat), max(10, len(flat) // 100))
            indices = np.random.choice(len(flat), n_params, replace=False)
            for idx in indices:
                flat[idx] += eps
                p.data = flat.reshape(p.data.shape)
                lp_c = _gather_log_probs(_model_log_probs(policy, chosen_np), chosen_np)
                lp_r = _gather_log_probs(_model_log_probs(policy, rejected_np), rejected_np)
                sft = -np.mean(lp_c)
                cp = np.exp(lp_c)
                rp = np.exp(lp_r)
                oc = cp / (1.0 - cp + 1e-8)
                orej = rp / (1.0 - rp + 1e-8)
                lor = np.log(oc / (orej + 1e-8) + 1e-8)
                ol = -np.mean(np.log(1.0 / (1.0 + np.exp(-lor)) + 1e-8))
                loss_p = sft + self.beta * ol
                flat_grad = np.zeros_like(flat)
                flat_grad[idx] = (loss_p - loss_val) / eps
                grad += flat_grad.reshape(grad.shape)
                flat[idx] = original.flatten()[idx]
            p.data = original
            p.grad = grad / n_params


class GRPOTrainer:
    def __init__(
        self,
        policy: Module,
        ref_model: Optional[Module] = None,
        beta: float = 0.1,
        gamma: float = 0.5,
        lr: float = 1e-6,
        num_generations: int = 4,
        clip_eps: float = 0.2,
        max_grad_norm: float = 1.0,
    ):
        self.policy = policy
        self.ref_model = ref_model
        self.beta = beta
        self.gamma = gamma
        self.num_generations = num_generations
        self.clip_eps = clip_eps
        self.max_grad_norm = max_grad_norm
        self.optimizer = Adam(policy.parameters(), lr=lr)

    def _normalize_rewards(self, rewards: np.ndarray) -> np.ndarray:
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8
        return (rewards - mean_r) / std_r

    def _compute_group_rewards(self, rewards: np.ndarray) -> np.ndarray:
        batch_size = rewards.shape[0] // self.num_generations
        normalized = np.zeros_like(rewards)
        for i in range(batch_size):
            group = rewards[i * self.num_generations:(i + 1) * self.num_generations]
            normalized[i * self.num_generations:(i + 1) * self.num_generations] = self._normalize_rewards(group)
        return normalized

    def train_step(
        self,
        prompts: Tensor,
        completions: Tensor,
        rewards: np.ndarray,
    ) -> float:
        self.policy.train()

        completions_np = completions.data.astype(np.int32) if isinstance(completions, Tensor) else np.array(completions, dtype=np.int32)

        normalized_rewards = self._compute_group_rewards(rewards)

        old_log_probs = _gather_log_probs(_model_log_probs(self.policy, completions_np), completions_np)
        new_log_probs = _gather_log_probs(_model_log_probs(self.policy, completions_np), completions_np)

        log_ratio = new_log_probs - old_log_probs
        ratio = np.exp(log_ratio)
        clipped = np.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        loss1 = ratio * normalized_rewards
        loss2 = clipped * normalized_rewards
        policy_loss = -float(np.minimum(loss1, loss2).mean())

        if self.ref_model is not None:
            ref_log_probs = _gather_log_probs(_model_log_probs(self.ref_model, completions_np), completions_np)
            kl = np.mean(new_log_probs - ref_log_probs)
            total_loss = policy_loss + self.beta * float(kl)
        else:
            total_loss = policy_loss

        self.policy.zero_grad()
        self._approximate_backward(self.policy, completions_np,
                                   old_log_probs, new_log_probs,
                                   normalized_rewards, total_loss)
        self.optimizer.step()

        return float(total_loss)

    def _approximate_backward(self, policy, completions_np,
                              old_log_probs, new_log_probs,
                              normalized_rewards, loss_val):
        eps = 1e-4
        for p in policy.parameters():
            if not p.requires_grad:
                continue
            grad = np.zeros_like(p.data)
            original = p.data.copy()
            flat = p.data.flatten()
            n_params = min(len(flat), max(10, len(flat) // 100))
            indices = np.random.choice(len(flat), n_params, replace=False)
            for idx in indices:
                flat[idx] += eps
                p.data = flat.reshape(p.data.shape)
                nlp = _gather_log_probs(_model_log_probs(policy, completions_np), completions_np)
                lr = np.exp(nlp - old_log_probs)
                cl = np.clip(lr, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                l1 = lr * normalized_rewards
                l2 = cl * normalized_rewards
                pl = -float(np.minimum(l1, l2).mean())
                if self.ref_model is not None:
                    rlp = _gather_log_probs(_model_log_probs(self.ref_model, completions_np), completions_np)
                    kl = np.mean(nlp - rlp)
                    loss_p = pl + self.beta * float(kl)
                else:
                    loss_p = pl
                flat_grad = np.zeros_like(flat)
                flat_grad[idx] = (loss_p - loss_val) / eps
                grad += flat_grad.reshape(grad.shape)
                flat[idx] = original.flatten()[idx]
            p.data = original
            p.grad = grad / n_params


class ConstitutionalAI:
    def __init__(
        self,
        model: Module,
        critique_model: Optional[Module] = None,
        lr: float = 1e-6,
        num_iterations: int = 3,
        temperature: float = 0.7,
    ):
        self.model = model
        self.critique_model = critique_model if critique_model is not None else model
        self.lr = lr
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.principles: List[str] = [
            "Is the response helpful and informative?",
            "Is the response harmless and safe?",
            "Is the response honest and accurate?",
            "Does the response avoid harmful stereotypes?",
            "Is the response clear and well-structured?",
        ]

    def add_principle(self, principle: str):
        self.principles.append(principle)

    def _generate_response(self, prompt: Tensor, max_tokens: int = 128) -> List[int]:
        if hasattr(self.model, 'generate'):
            with no_grad():
                return self.model.generate(prompt, max_tokens=max_tokens, temperature=self.temperature)
        logits = self.model(prompt)
        return [int(logits.data[0, -1].argmax())]

    def _generate_critique(self, prompt: Tensor, response_tokens: List[int], principle: str) -> List[int]:
        return self._generate_response(prompt, max_tokens=64)

    def _generate_revision(self, prompt: Tensor, response_tokens: List[int], critique_tokens: List[int]) -> List[int]:
        return self._generate_response(prompt, max_tokens=128)

    def _compute_sft_loss(self, prompt: Tensor, response_tokens: List[int]) -> float:
        input_ids = np.array([[prompt.data.flatten()[0]] + response_tokens], dtype=np.int32)
        targets = np.array(response_tokens + [0], dtype=np.int32)
        logits = _model_logits(self.model, input_ids)
        log_probs = _log_softmax_np(logits, axis=-1)
        token_log_probs = log_probs[0, np.arange(len(targets)), targets]
        return -float(token_log_probs.mean())

    def revise(self, prompt: Tensor, max_tokens: int = 128) -> Tuple[List[int], List[float]]:
        response = self._generate_response(prompt, max_tokens)
        losses = []
        current_response = response

        for i in range(self.num_iterations):
            principle = self.principles[i % len(self.principles)]
            critique = self._generate_critique(prompt, current_response, principle)
            revised = self._generate_revision(prompt, current_response, critique)
            current_response = revised

        loss = self._compute_sft_loss(prompt, current_response)
        losses.append(loss)

        return current_response, losses


class RejectionSamplingFT:
    def __init__(
        self,
        model: Module,
        lr: float = 1e-5,
        num_samples: int = 16,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ):
        self.model = model
        self.lr = lr
        self.num_samples = num_samples
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.optimizer = Adam(model.parameters(), lr=lr)

    def _generate_samples(self, prompt: Tensor, max_tokens: int = 128) -> List[List[int]]:
        samples = []
        if hasattr(self.model, 'generate'):
            with no_grad():
                for _ in range(self.num_samples):
                    sample = self.model.generate(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=self.temperature,
                        top_k=self.top_k,
                        top_p=self.top_p,
                    )
                    samples.append(sample)
        else:
            for _ in range(self.num_samples):
                samples.append([])
        return samples

    def _filter_correct(
        self,
        prompt: Tensor,
        samples: List[List[int]],
        verify_fn: Any = None,
    ) -> List[List[int]]:
        if verify_fn is None:
            return samples[:1]
        return [s for s in samples if verify_fn(prompt, s)]

    def _sft_loss(self, prompt: Tensor, tokens: List[int]) -> float:
        input_ids = np.array([[prompt.data.flatten()[0]] + tokens], dtype=np.int32)
        targets = np.array(tokens + [0], dtype=np.int32)
        logits = _model_logits(self.model, input_ids)
        log_probs = _log_softmax_np(logits, axis=-1)
        token_log_probs = log_probs[0, np.arange(len(targets)), targets]
        return -float(token_log_probs.mean())

    def train(
        self,
        prompts: List[Tensor],
        max_tokens: int = 128,
        verify_fn: Any = None,
        num_epochs: int = 1,
    ) -> List[float]:
        self.model.train()
        all_losses = []

        for epoch in range(num_epochs):
            for prompt in prompts:
                samples = self._generate_samples(prompt, max_tokens)
                correct_samples = self._filter_correct(prompt, samples, verify_fn)

                if not correct_samples:
                    continue

                for correct_sample in correct_samples:
                    loss_val = self._sft_loss(prompt, correct_sample)
                    all_losses.append(loss_val)

        return all_losses

    def best_of_k(
        self,
        prompt: Tensor,
        max_tokens: int = 128,
        verify_fn: Any = None,
    ) -> Tuple[List[int], float]:
        samples = self._generate_samples(prompt, max_tokens)
        if verify_fn is not None:
            correct = [s for s in samples if verify_fn(prompt, s)]
            if correct:
                return correct[0], 1.0
        return samples[0] if samples else [], 0.0
