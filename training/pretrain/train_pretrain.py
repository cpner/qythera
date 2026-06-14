import os
import math
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from vaelon.config import VaelonConfig
from vaelon.model import VaelonModel


@dataclass
class PretrainConfig:
    model_name: str = "vaelon-7b"
    output_dir: str = "./outputs/pretrain"
    num_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    max_seq_len: int = 2048
    bf16: bool = True
    logging_steps: int = 10
    save_steps: int = 1000
    eval_steps: int = 500
    wandb_project: Optional[str] = None
    deepspeed_config: Optional[str] = None


class PretrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = 2048):
        self.data = []
        self.max_length = max_length
        if os.path.exists(data_path):
            import json
            with open(data_path) as f:
                self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        ids = sample.get("input_ids", [])[:self.max_length]
        labels = sample.get("labels", ids.copy())
        attention_mask = [1] * len(ids) + [0] * (self.max_length - len(ids))
        ids = ids + [0] * (self.max_length - len(ids))
        labels = labels + [-100] * (self.max_length - len(labels))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(config: PretrainConfig):
    print(f"Starting pretraining: {config.model_name}")
    os.makedirs(config.output_dir, exist_ok=True)

    vaelon_config = VaelonConfig.vaelon_7b()
    model = VaelonModel(vaelon_config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e9:.2f}B)")

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    dataset = PretrainDataset("data/tokenized/train.json", config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    num_update_steps = (len(dataloader) * config.num_epochs) // config.gradient_accumulation_steps
    num_warmup_steps = int(num_update_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_update_steps)

    if config.wandb_project and HAS_WANDB:
        wandb.init(project=config.wandb_project, name=config.model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    global_step = 0
    for epoch in range(config.num_epochs):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()
            total_loss += loss.item()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % config.logging_steps == 0:
                    avg_loss = total_loss / config.logging_steps
                    lr = scheduler.get_last_lr()[0]
                    print(f"Step {global_step} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
                    if config.wandb_project and HAS_WANDB:
                        wandb.log({"loss": avg_loss, "lr": lr, "step": global_step})
                    total_loss = 0.0

                if global_step % config.save_steps == 0:
                    save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    torch.save({"model": model.state_dict(), "step": global_step}, os.path.join(save_path, "model.pt"))
                    print(f"Saved checkpoint to {save_path}")

    final_path = os.path.join(config.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_path, "model.pt"))
    print(f"Training complete! Final model saved to {final_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Vaelon Pretraining")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--deepspeed", type=str, default=None)
    args = parser.parse_args()
    config = PretrainConfig()
    if args.deepspeed:
        config.deepspeed_config = args.deepspeed
    train(config)
