import os
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from vaelon.config import VaelonConfig
from vaelon.model import VaelonModel


@dataclass
class DPOConfig:
    beta: float = 0.1
    learning_rate: float = 5e-7
    output_dir: str = "./outputs/dpo"
    num_epochs: int = 1
    batch_size: int = 2
    max_seq_len: int = 2048
    lora_enabled: bool = False
    reference_model_path: Optional[str] = None


class DPOTrainer:
    def __init__(self, config: DPOConfig, model: VaelonModel, ref_model: Optional[VaelonModel] = None):
        self.config = config
        self.model = model
        self.ref_model = ref_model or self._create_ref_model()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    def _create_ref_model(self):
        ref = VaelonModel(self.model.config)
        ref.load_state_dict(self.model.state_dict())
        for p in ref.parameters():
            p.requires_grad = False
        return ref

    def compute_log_probs(self, model, input_ids, attention_mask, labels):
        with torch.no_grad() if model == self.ref_model else torch.enable_grad():
            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]
            target = labels[:, 1:]
            log_probs = F.log_softmax(logits, dim=-1)
            per_token_log_probs = torch.gather(log_probs, 2, target.unsqueeze(-1)).squeeze(-1)
            mask = (target != -100).float()
            seq_log_probs = (per_token_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        return seq_log_probs

    def dpo_loss(self, chosen_logps, rejected_logps):
        loss = -F.logsigmoid(self.config.beta * (chosen_logps - rejected_logps)).mean()
        chosen_rewards = self.config.beta * chosen_logps.detach()
        rejected_rewards = self.config.beta * rejected_logps.detach()
        reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()
        reward_margin = (chosen_rewards - rejected_rewards).mean()
        return loss, {
            "reward_accuracy": reward_accuracy.item(),
            "reward_margin": reward_margin.item(),
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
        }

    def train_step(self, batch):
        self.model.train()
        chosen_ids = batch["chosen_input_ids"]
        chosen_mask = batch["chosen_attention_mask"]
        chosen_labels = batch["chosen_labels"]
        rejected_ids = batch["rejected_input_ids"]
        rejected_mask = batch["rejected_attention_mask"]
        rejected_labels = batch["rejected_labels"]

        chosen_logps = self.compute_log_probs(self.model, chosen_ids, chosen_mask, chosen_labels)
        with torch.no_grad():
            ref_chosen_logps = self.compute_log_probs(self.ref_model, chosen_ids, chosen_mask, chosen_labels)
            ref_rejected_logps = self.compute_log_probs(self.ref_model, rejected_ids, rejected_mask, rejected_labels)

        chosen_logps = chosen_logps - ref_chosen_logps
        rejected_logps = self.compute_log_probs(self.model, rejected_ids, rejected_mask, rejected_labels) - ref_rejected_logps

        loss, metrics = self.dpo_loss(chosen_logps, rejected_logps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item(), metrics

    def train(self, dataloader, num_epochs=1):
        for epoch in range(num_epochs):
            for step, batch in enumerate(dataloader):
                loss, metrics = self.train_step(batch)
                if step % 10 == 0:
                    print(f"Epoch {epoch} Step {step} | Loss: {loss:.4f} | Acc: {metrics['reward_accuracy']:.3f}")

        os.makedirs(self.config.output_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.config.output_dir, "model.pt"))
        print(f"DPO training complete! Saved to {self.config.output_dir}")
