
import torch, os, json, time
from torch.utils.data import Dataset, DataLoader
from core.config import TrainingConfig, Config
from core.model import VaelonModel
from core.tokenizer import Tokenizer

class ChatDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=2048):
        self.samples, self.max_len = [], max_len
        if os.path.exists(path):
            with open(path) as f: data = json.load(f)
            for item in data:
                msgs = item.get("messages", [])
                ids = tokenizer.encode_chat(msgs)
                if len(ids) < max_len:
                    self.samples.append(ids)

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        ids = self.samples[i]
        pad_len = self.max_len - len(ids)
        ids = ids + [2] * pad_len
        labels = ids.copy()
        labels[-pad_len:] = [-100] * pad_len
        mask = [1]*(len(ids)-pad_len) + [0]*pad_len
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(mask)}

class Trainer:
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VaelonModel(self.config.model)
        self.model.to(self.device)
        self.tokenizer = Tokenizer()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

    def train(self, data_path, epochs=1):
        print(f"Training on {self.device} | {sum(p.numel() for p in self.model.parameters())/1e6:.0f}M params")
        os.makedirs(self.config.output_dir, exist_ok=True)
        dataset = ChatDataset(data_path, self.tokenizer, self.config.model.max_seq_len)
        if len(dataset) == 0:
            print("No training data. Create data/chat_train.json")
            return
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        self.model.train()
        step = 0
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                _, loss = self.model(ids, labels)
                loss = loss / self.config.gradient_accumulation
                loss.backward()
                total_loss += loss.item()
                if (step + 1) % self.config.gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                step += 1
                if step % self.config.log_steps == 0:
                    print(f"  Step {step} | Loss: {total_loss:.4f}")
                    total_loss = 0
                if step % self.config.save_steps == 0:
                    self._save(f"step_{step}")
        self._save("final")
        print("Training complete!")

    def _save(self, name):
        path = os.path.join(self.config.output_dir, name)
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
        print(f"  Saved: {path}")

if __name__ == "__main__":
    import sys
    t = Trainer()
    data = sys.argv[1] if len(sys.argv) > 1 else "data/chat_train.json"
    t.train(data)
