import os
import json
from dataclasses import dataclass
from typing import Optional, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    HAS_HF = True
except ImportError:
    HAS_HF = False

from vaelon.config import VaelonConfig
from vaelon.model import VaelonModel


@dataclass
class SFTConfig:
    model_path: Optional[str] = None
    output_dir: str = "./outputs/sft"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_seq_len: int = 2048
    bf16: bool = True
    lora_enabled: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None
    response_loss_only: bool = True


class ChatDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048,
                 response_loss_only: bool = True):
        self.samples = []
        self.max_length = max_length
        self.response_loss_only = response_loss_only
        self.eos_token_id = tokenizer.eos_token_id or 0
        self.pad_token_id = tokenizer.pad_token_id or 0

        with open(data_path) as f:
            raw_data = json.load(f)

        for item in raw_data:
            messages = item.get("messages", [])
            if not messages:
                continue
            input_ids = []
            labels = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                tokens = tokenizer.encode(f"<|{role}|>\n{content}", add_special_tokens=False)
                input_ids.extend(tokens)
                if response_loss_only and role == "assistant":
                    labels.extend(tokens)
                elif not response_loss_only:
                    labels.extend(tokens)
                else:
                    labels.extend([-100] * len(tokens))
            input_ids.append(self.eos_token_id)
            labels.append(self.eos_token_id)
            if len(input_ids) <= max_length:
                pad_len = max_length - len(input_ids)
                input_ids += [self.pad_token_id] * pad_len
                labels += [-100] * pad_len
                self.samples.append({
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": [1] * (len(input_ids) - pad_len) + [0] * pad_len,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "input_ids": torch.tensor(s["input_ids"], dtype=torch.long),
            "labels": torch.tensor(s["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(s["attention_mask"], dtype=torch.long),
        }


def train_sft(config: SFTConfig):
    print("Starting SFT training...")
    os.makedirs(config.output_dir, exist_ok=True)

    if config.lora_target_modules is None:
        config.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    vaelon_config = VaelonConfig.vaelon_7b()
    model = VaelonModel(vaelon_config)

    if config.lora_enabled and HAS_PEFT:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    from vaelon.tokenizer import VaelonTokenizer
    tokenizer = VaelonTokenizer()

    dataset = ChatDataset("data/chat_train.json", tokenizer, config.max_seq_len, config.response_loss_only)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    global_step = 0
    for epoch in range(config.num_epochs):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()
            total_loss += loss.item()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    print(f"Epoch {epoch} Step {global_step} | Loss: {total_loss:.4f}")
                    total_loss = 0.0

    save_path = os.path.join(config.output_dir, "final")
    os.makedirs(save_path, exist_ok=True)
    if config.lora_enabled and HAS_PEFT:
        model.save_pretrained(save_path)
    else:
        torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
    print(f"SFT training complete! Saved to {save_path}")


if __name__ == "__main__":
    config = SFTConfig()
    train_sft(config)
