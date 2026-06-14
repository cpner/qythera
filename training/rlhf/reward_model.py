import torch
import torch.nn as nn
from typing import Optional

from vaelon.config import VaelonConfig
from vaelon.model import VaelonModel


class RewardModel(nn.Module):
    def __init__(self, base_model: VaelonModel):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.hidden_states
        last_token_mask = attention_mask.sum(dim=-1) - 1
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        last_hidden = hidden[batch_idx, last_token_mask]
        reward = self.reward_head(last_hidden).squeeze(-1)
        return reward

    def compute_loss(self, chosen_ids, chosen_mask, rejected_ids, rejected_mask):
        chosen_rewards = self.forward(chosen_ids, chosen_mask)
        rejected_rewards = self.forward(rejected_ids, rejected_mask)
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
        accuracy = (chosen_rewards > rejected_rewards).float().mean()
        return loss, {"accuracy": accuracy.item(), "chosen_reward": chosen_rewards.mean().item(), "rejected_reward": rejected_rewards.mean().item()}
