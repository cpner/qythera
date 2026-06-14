import torch
import torch.nn.functional as F
from typing import Optional, Dict

from vaelon.config import VaelonConfig
from vaelon.model import VaelonModel


class PPOTrainer:
    def __init__(self, model, ref_model=None, reward_model=None, lr=1e-5, clip_range=0.2, gamma=0.99, lam=0.95):
        self.model = model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.clip_range = clip_range
        self.gamma = gamma
        self.lam = lam

    def compute_gae(self, rewards, values, dones, last_value):
        advantages = []
        gae = 0
        values = values + [last_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [a + v for a, v in zip(advantages, values[:-1])]
        return advantages, returns

    def ppo_loss(self, old_logprobs, new_logprobs, advantages, clip_range):
        ratio = torch.exp(new_logprobs - old_logprobs)
        clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        loss1 = -advantages * ratio
        loss2 = -advantages * clipped_ratio
        return torch.max(loss1, loss2).mean()

    def kl_penalty(self, logprobs, ref_logprobs):
        return (logprobs - ref_logprobs).mean()

    def train_step(self, prompts, max_new_tokens=256):
        self.model.train()
        with torch.no_grad():
            outputs = self.model.generate(prompts, max_new_tokens=max_new_tokens)
            if self.reward_model:
                rewards = self.reward_model(outputs).tolist()
            else:
                rewards = [0.0] * len(outputs)

        metrics = {"mean_reward": sum(rewards) / len(rewards)}
        return metrics
