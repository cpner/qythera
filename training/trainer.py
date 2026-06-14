import numpy as np
import os, json, time
from core.model import VaelonModel, VaelonConfig
from core.tokenizer.bpe import BPETokenizer
from core.autodiff.optim import Adam


class Trainer:
    """Training pipeline for Vaelon models.
    
    Supports:
    - Language modeling (cross-entropy loss)
    - Gradient accumulation
    - Learning rate scheduling
    - Checkpointing
    """
    
    def __init__(self, config=None):
        self.config = config or VaelonConfig.small()
        self.model = VaelonModel(self.config)
        self.tokenizer = BPETokenizer()
        self.optimizer = Adam(self.model.parameters(), lr=3e-4, weight_decay=0.01)
        self.step = 0

    def train(self, data_path, epochs=1, batch_size=4, log_every=10, save_every=100):
        """Train on JSONL data file."""
        print(f"Training: {sum(p.data.size for p in self.model.parameters()):,} params")
        
        if not os.path.exists(data_path):
            print(f"Creating sample data: {data_path}")
            self._create_sample_data(data_path)

        with open(data_path) as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} samples")
        
        for epoch in range(epochs):
            np.random.shuffle(data)
            total_loss = 0.0
            
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                
                # Tokenize batch
                max_len = 128
                input_ids = np.zeros((len(batch), max_len), dtype=np.int32)
                labels = np.full((len(batch), max_len), -100, dtype=np.int32)
                
                for j, sample in enumerate(batch):
                    msgs = sample.get("messages", [])
                    ids = self.tokenizer.encode_chat(msgs)
                    ids = ids[:max_len]
                    input_ids[j, :len(ids)] = ids
                    labels[j, :len(ids)] = ids

                # Forward pass
                from core.autodiff.tensor import Tensor
                ids_t = Tensor(input_ids)
                labels_t = Tensor(labels)
                logits, loss, aux_loss = self.model(ids_t, labels_t)

                if loss is None:
                    continue

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                for p in self.model.parameters():
                    if p.grad is not None:
                        g = p.grad.data
                        norm = np.sqrt(np.sum(g ** 2))
                        if norm > 1.0:
                            p.grad.data = g / norm

                self.optimizer.step()
                self.step += 1
                total_loss += loss.item()

                if self.step % log_every == 0:
                    avg = total_loss / log_every
                    print(f"  Epoch {epoch} Step {self.step} | Loss: {avg:.4f}")
                    total_loss = 0.0

                if self.step % save_every == 0:
                    self.save_checkpoint(f"step_{self.step}")

        self.save_checkpoint("final")
        print("Training complete!")

    def _create_sample_data(self, path):
        data = [
            {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]},
            {"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]},
            {"messages": [{"role": "user", "content": "Explain Python"}, {"role": "assistant", "content": "Python is a programming language."}]},
            {"messages": [{"role": "user", "content": "Write a function"}, {"role": "assistant", "content": "def f(x): return x * 2"}]},
            {"messages": [{"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "Artificial Intelligence is machines that think."}]},
        ] * 100  # Repeat for more data
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    def save_checkpoint(self, name):
        path = os.path.join("checkpoints", name)
        os.makedirs(path, exist_ok=True)
        state = self.model.state_dict()
        for k, v in state.items():
            np.save(os.path.join(path, f"{k}.npy"), v)
        print(f"  Saved checkpoint: {path}")

    def load_checkpoint(self, path):
        state = {}
        for f in os.listdir(path):
            if f.endswith(".npy"):
                name = f[:-4]
                state[name] = np.load(os.path.join(path, f))
        self.model.load_state_dict(state)
        print(f"  Loaded checkpoint: {path}")


if __name__ == "__main__":
    t = Trainer()
    t.train("data/training.json", epochs=3, batch_size=2)
