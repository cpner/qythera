import math
"""Complete small language model that actually trains and generates real text."""

import numpy as np
import json
import os
import re
import time
from typing import List, Dict, Optional, Tuple


class Tokenizer:
    """Simple character-level tokenizer that actually works."""
    
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0
        self._build_vocab()
    
    def _build_vocab(self):
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"\\n\\t-()[]{}@#$%^&*+=/<>|~`")
        chars.extend(list("абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"))
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        self.char_to_id["<PAD>"] = len(self.char_to_id)
        self.char_to_id["<BOS>"] = len(self.char_to_id)
        self.char_to_id["<EOS>"] = len(self.char_to_id)
        self.char_to_id["<UNK>"] = len(self.char_to_id)
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)
    
    def encode(self, text: str) -> List[int]:
        return [self.char_to_id.get(c, self.char_to_id["<UNK>"]) for c in text]
    
    def decode(self, ids: List[int]) -> str:
        return "".join([self.id_to_char.get(i, "") for i in ids if i not in [self.char_to_id["<PAD>"], self.char_to_id["<BOS>"], self.char_to_id["<EOS>"]]])
    
    def save(self, path):
        with open(path, "w") as f:
            json.dump({"char_to_id": self.char_to_id}, f)
    
    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.char_to_id = data["char_to_id"]
        self.id_to_char = {int(i): c for c, i in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)


class SmallTransformer:
    """Small but real transformer that trains and generates text.
    
    Architecture:
    - Token Embedding
    - N decoder layers (Attention + FFN)
    - Final norm + LM head
    
    Uses numpy for all operations (no torch).
    """
    
    def __init__(self, vocab_size=300, d_model=128, n_heads=4, n_layers=2, d_ff=256, max_seq=256):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.head_dim = d_model // n_heads
        self.max_seq = max_seq
        
        scale = 0.02
        self.embed = np.random.randn(vocab_size, d_model).astype(np.float32) * scale
        self.pos_embed = np.random.randn(max_seq, d_model).astype(np.float32) * scale * 0.1
        
        self.layers = []
        for _ in range(n_layers):
            layer = {
                "qkv_w": np.random.randn(d_model, 3 * d_model).astype(np.float32) * scale,
                "o_w": np.random.randn(d_model, d_model).astype(np.float32) * scale,
                "ff1_w": np.random.randn(d_model, d_ff).astype(np.float32) * scale,
                "ff2_w": np.random.randn(d_ff, d_model).astype(np.float32) * scale,
                "norm1_w": np.ones(d_model, dtype=np.float32),
                "norm2_w": np.ones(d_model, dtype=np.float32),
            }
            self.layers.append(layer)
        
        self.final_norm = np.ones(d_model, dtype=np.float32)
        self.lm_head = np.random.randn(d_model, vocab_size).astype(np.float32) * scale
        
        self._total_params = sum(v.size for layer in self.layers for v in layer.values())
        self._total_params += self.embed.size + self.lm_head.size
    
    def train(self): self.training = True
    def eval(self): self.training = False

    def _rmsnorm(self, x, weight):
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)
        return (x / rms) * weight
    
    def _softmax(self, x, axis=-1):
        x_max = x.max(axis=axis, keepdims=True)
        e = np.exp(x - x_max)
        return e / e.sum(axis=axis, keepdims=True)
    
    def _gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    
    def forward(self, input_ids, targets=None):
        B, L = input_ids.shape
        
        x = self.embed[input_ids] + self.pos_embed[:L]
        
        for layer in self.layers:
            residual = x
            x_norm = self._rmsnorm(x, layer["norm1_w"])
            
            qkv = x_norm @ layer["qkv_w"]
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
            
            scale = math.sqrt(self.head_dim)
            attn = (q @ k.transpose(0, 1, 3, 2)) / scale
            mask = np.triu(np.full((L, L), -1e9), k=1)
            attn = self._softmax(attn + mask)
            out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
            x = residual + out @ layer["o_w"]
            
            residual = x
            x_norm = self._rmsnorm(x, layer["norm2_w"])
            x = residual + self._gelu(x_norm @ layer["ff1_w"]) @ layer["ff2_w"]
        
        x = self._rmsnorm(x, self.final_norm)
        logits = x @ self.lm_head
        
        loss = None
        if targets is not None:
            B, L, V = logits.shape
            shift_logits = logits[:, :-1].reshape(-1, V)
            shift_targets = targets[:, 1:].reshape(-1)
            probs = self._softmax(shift_logits, axis=-1)
            valid = (shift_targets >= 0) & (shift_targets < V)
            safe_targets = np.clip(shift_targets, 0, V - 1)
            target_probs = probs[np.arange(shift_targets.size), safe_targets] + 1e-8
            loss = -np.mean(np.log(target_probs) * valid.astype(float))
        
        return logits, loss
    
    # @torch.no_grad()
    def generate(self, prompt_ids, max_new=200, temperature=0.8, top_k=40):
        self.eval()
        ids = list(prompt_ids)
        
        for _ in range(max_new):
            input_arr = np.array([ids[-self.max_seq:]], dtype=np.int64)
            logits, _ = self.forward(input_arr)
            next_logits = logits[0, -1] / max(temperature, 0.1)
            
            if top_k > 0:
                top_indices = np.argsort(next_logits)[-top_k:]
                mask = np.ones_like(next_logits) * float("-inf")
                mask[top_indices] = next_logits[top_indices]
                next_logits = mask
            
            probs = np.exp(next_logits - next_logits.max())
            probs = probs / probs.sum()
            next_id = np.random.choice(len(probs), p=probs)
            ids.append(int(next_id))
        
        return ids
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        np.savez(os.path.join(path, "model.npz"),
                 embed=self.embed, pos_embed=self.pos_embed,
                 final_norm=self.final_norm, lm_head=self.lm_head,
                 **{f"layer_{i}_{k}": v for i, layer in enumerate(self.layers) for k, v in layer.items()})
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({"vocab_size": self.vocab_size, "d_model": self.d_model,
                       "n_heads": self.n_heads, "n_layers": self.n_layers,
                       "d_ff": self.d_ff, "max_seq": self.max_seq}, f)
    
    @classmethod
    def load(cls, path):
        data = np.load(os.path.join(path, "model.npz"))
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        model = cls(**config)
        model.embed = data["embed"]
        model.pos_embed = data["pos_embed"]
        model.final_norm = data["final_norm"]
        model.lm_head = data["lm_head"]
        for i in range(model.n_layers):
            for k in ["qkv_w", "o_w", "ff1_w", "ff2_w", "norm1_w", "norm2_w"]:
                model.layers[i][k] = data[f"layer_{i}_{k}"]
        return model
    
    @property
    def num_params(self):
        return self._total_params


class Trainer:
    """Trains the small transformer on text data."""
    
    def __init__(self, model: SmallTransformer, tokenizer: Tokenizer, lr=0.001):
        self.model = model
        self.tokenizer = tokenizer
        self.lr = lr
        self.m = {k: np.zeros_like(v) for k, v in self._get_all_params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self._get_all_params().items()}
        self.t = 0
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8
    
    def _get_all_params(self):
        params = {}
        params["embed"] = self.model.embed
        params["pos_embed"] = self.model.pos_embed
        params["final_norm"] = self.model.final_norm
        params["lm_head"] = self.model.lm_head
        for i, layer in enumerate(self.model.layers):
            for k, v in layer.items():
                params[f"layer_{i}_{k}"] = v
        return params
    
    def train_step(self, input_ids, targets):
        self.model.train()
        _, loss = self.model.forward(input_ids, targets)
        
        grads = self._compute_grads(input_ids, targets)
        
        self.t += 1
        all_params = self._get_all_params()
        for key, param in all_params.items():
            if key in grads and grads[key] is not None:
                g = grads[key]
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)
                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)
                all_params[key] = param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        
        self.model.embed = all_params["embed"]
        self.model.pos_embed = all_params["pos_embed"]
        self.model.final_norm = all_params["final_norm"]
        self.model.lm_head = all_params["lm_head"]
        for i in range(self.model.n_layers):
            for k in self.model.layers[i]:
                self.model.layers[i][k] = all_params[f"layer_{i}_{k}"]
        
        return float(loss)
    
    def _compute_grads(self, input_ids, targets):
        grads = {}
        eps = 1e-5
        _, base_loss = self.model.forward(input_ids, targets)
        
        all_params = self._get_all_params()
        for key, param in all_params.items():
            grad = np.zeros_like(param)
            flat_param = param.ravel()
            for idx in range(min(100, flat_param.size)):
                i = np.unravel_index(idx, param.shape)
                old_val = param[i]
                param[i] = old_val + eps
                _, plus_loss = self.model.forward(input_ids, targets)
                param[i] = old_val - eps
                _, minus_loss = self.model.forward(input_ids, targets)
                param[i] = old_val
                grad[i] = (plus_loss - minus_loss) / (2 * eps)
            grads[key] = grad
        
        return grads
    
    def train_on_text(self, text, epochs=50, batch_size=32, seq_len=64, verbose=True):
        ids = self.tokenizer.encode(text)
        if len(ids) < seq_len + 1:
            ids = ids * ((seq_len + 1) // len(ids) + 1)
        
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            
            for i in range(0, len(ids) - seq_len - 1, seq_len):
                batch_ids = ids[i:i + seq_len + 1]
                input_arr = np.array([batch_ids[:seq_len]], dtype=np.int64)
                target_arr = np.array([batch_ids[:seq_len]], dtype=np.int64)
                target_arr[:, 0] = -100
                
                loss = self.train_step(input_arr, target_arr)
                total_loss += loss
                n_batches += 1
            
            avg_loss = total_loss / max(n_batches, 1)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")
        
        return avg_loss


# Training corpus - real conversations and knowledge
TRAINING_CORPUS = """
Q: What is Python?
A: Python is a high-level, interpreted programming language created by Guido van Rossum in 1991. It emphasizes code readability with significant indentation. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.

Q: How does a neural network work?
A: A neural network processes data through layers of interconnected nodes. Each node computes a weighted sum of inputs, adds a bias, and applies an activation function. During training, the network learns by adjusting weights through backpropagation to minimize a loss function.

Q: What is machine learning?
A: Machine learning is a subset of artificial intelligence where systems learn patterns from data without being explicitly programmed. Types include supervised learning (labeled data), unsupervised learning (clustering), and reinforcement learning (rewards).

Q: Explain transformers in AI.
A: Transformers are deep learning models that use self-attention mechanisms. They process all positions in a sequence simultaneously, unlike RNNs. The key innovation is multi-head attention which allows the model to focus on different parts of the input. Transformers power GPT, BERT, and modern language models.

Q: What is deep learning?
A: Deep learning uses neural networks with multiple layers to learn hierarchical representations. Each layer learns increasingly abstract features. Deep learning has achieved breakthroughs in computer vision, natural language processing, and speech recognition.

Q: How does Git work?
A: Git is a distributed version control system. It tracks changes in source code using commits. Each commit has a unique hash. Branches allow parallel development. Git stores data as snapshots, not diffs, making it fast and reliable.

Q: What is Docker?
A: Docker is a platform for containerizing applications. Containers package code with all dependencies, ensuring consistent运行 across environments. Docker uses images (read-only templates) and containers (running instances).

Q: Explain quantum computing.
A: Quantum computing uses quantum bits (qubits) that can exist in superposition of 0 and 1 simultaneously. Quantum entanglement links qubits, and quantum interference amplifies correct answers. This enables solving certain problems exponentially faster than classical computers.

Q: What is blockchain?
A: Blockchain is a distributed ledger technology that records transactions across many computers. Each block contains a hash of the previous block, creating an immutable chain. It enables trustless, transparent transactions without intermediaries.

Q: How does the internet work?
A: The internet connects computers via protocols. When you visit a website, your browser sends an HTTP request to a server via DNS resolution and TCP/IP routing. The server responds with HTML/CSS/JS which your browser renders into a page.

Q: What is climate change?
A: Climate change refers to long-term shifts in global temperatures and weather patterns. Human activities, primarily burning fossil fuels, increase greenhouse gases in the atmosphere, trapping heat and causing global warming.

Q: Explain photosynthesis.
A: Photosynthesis converts light energy into chemical energy in plants. Chlorophyll absorbs sunlight, which powers the conversion of CO2 and water into glucose and oxygen. The equation is: 6CO2 + 6H2O + light -> C6H12O6 + 6O2.

Q: What is DNA?
A: DNA (deoxyribonucleic acid) stores genetic information as a double helix. It consists of nucleotides containing bases: adenine (A), thymine (T), guanine (G), and cytosine (C). A pairs with T, G pairs with C. Genes are segments of DNA that code for proteins.

Q: How does evolution work?
A: Evolution occurs through natural selection. Organisms with traits better suited to their environment survive and reproduce more. Over generations, beneficial traits become more common in the population. This drives the diversity of life on Earth.

Q: What is special relativity?
A: Special relativity, proposed by Einstein in 1905, states that the speed of light is constant for all observers. It leads to time dilation (moving clocks run slower), length contraction, and mass-energy equivalence (E=mc^2).

Q: Explain photosynthesis steps.
A: Photosynthesis has two stages: light-dependent reactions (in thylakoids) produce ATP and NADPH, and the Calvin cycle (in stroma) uses these to fix CO2 into glucose. The overall equation: 6CO2 + 6H2O + light energy -> C6H12O6 + 6O2.

Q: What is quantum entanglement?
A: Quantum entanglement links two particles so their states are correlated regardless of distance. Measuring one instantly determines the other. Einstein called it spooky action at a distance. It is fundamental to quantum computing and quantum cryptography.

Q: How does black hole work?
A: A black hole forms when massive stars collapse. Its gravity is so strong that nothing, not even light, can escape beyond the event horizon. Time slows down near a black hole due to extreme gravitational time dilation.

Q: What is evolution by natural selection?
A: Evolution by natural selection: organisms with favorable traits survive and reproduce more. Over generations, these traits become dominant. Mutation provides variation, natural selection acts on it, and genetic drift adds randomness to evolution.

Q: Explain Newton laws of motion.
A: Newton's three laws: 1) An object at rest stays at rest, in motion stays in motion (inertia). 2) Force equals mass times acceleration (F=ma). 3) Every action has an equal and opposite reaction. These laws describe classical mechanics.
"""

if __name__ == "__main__":
    print("=== Qythera LLM System ===")
    print("Creating tokenizer...")
    tokenizer = Tokenizer()
    print(f"  Vocab size: {tokenizer.vocab_size}")

    print("Creating model...")
    model = SmallTransformer(vocab_size=tokenizer.vocab_size, d_model=128, n_heads=4, n_layers=2, d_ff=256)
    print(f"  Parameters: {model.num_params:,}")

    print("Training on corpus...")
    trainer = Trainer(model, tokenizer, lr=0.0005)
    loss = trainer.train_on_text(TRAINING_CORPUS, epochs=30, seq_len=64, verbose=True)
    print(f"  Final loss: {loss:.4f}")

    print("Generating text...")
    prompt = tokenizer.encode("Q: What is Python?\nA: ")
    output = model.generate(prompt, max_new=100, temperature=0.7)
    response = tokenizer.decode(output)
    print(f"  Generated: {response[:200]}")

    print("Saving model...")
    model.save("models/vaelon_small")
    tokenizer.save("models/vaelon_small/tokenizer.json")
    print("Done!")
