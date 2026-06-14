"""Complete training pipeline with numerical gradients."""

import numpy as np
import time
import json
import os
from typing import Optional, List
from core.transformer.model import VaelonTransformer, ModelConfig


class Trainer:
    def __init__(self, model, lr=0.001, weight_decay=0.01):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.step_count = 0
        self.best_loss = float('inf')
        self._setup_optimizer()
    
    def _setup_optimizer(self):
        self.m = {}
        self.v = {}
        for name, param in self._get_params():
            self.m[name] = np.zeros_like(param)
            self.v[name] = np.zeros_like(param)
        self.t = 0
    
    def _get_params(self):
        params = [("embed", self.model.embed.weight), ("lm_head", self.model.lm_head)]
        for i, layer in enumerate(self.model.layers):
            for k in ["attn_norm.weight", "ffn_norm.weight"]:
                params.append((f"L{i}.{k.split('.')[0]}", getattr(layer, k.split('.')[0]).weight))
            for k in ["wq", "wk", "wv", "wo"]:
                params.append((f"L{i}.attn.{k}", getattr(layer.attn, k)))
            for k in ["w1", "w2", "w3"]:
                if hasattr(layer.ffn, k):
                    params.append((f"L{i}.ffn.{k}", getattr(layer.ffn, k)))
        return params
    
    def compute_loss(self, input_ids, targets):
        logits, _ = self.model.forward(input_ids)
        B, L, V = logits.shape
        shift_logits = logits[:, :-1].reshape(-1, V)
        shift_targets = targets[:, 1:].reshape(-1)
        probs = np.exp(shift_logits - shift_logits.max(axis=-1, keepdims=True))
        probs = probs / (probs.sum(axis=-1, keepdims=True) + 1e-8)
        valid = (shift_targets >= 0) & (shift_targets < V)
        safe_targets = np.clip(shift_targets, 0, V - 1)
        target_probs = probs[np.arange(shift_targets.size), safe_targets] + 1e-8
        return -np.mean(np.log(target_probs) * valid.astype(float))
    
    def compute_gradients(self, input_ids, targets, num_samples=30):
        grads = {}
        all_params = self._get_params()
        param_keys = [name for name, _ in all_params]
        sample_indices = np.random.choice(len(param_keys), min(num_samples, len(param_keys)), replace=False)
        eps = 1e-4
        for idx in sample_indices:
            name = param_keys[idx]
            param = all_params[idx][1]
            flat = param.ravel()
            grad_flat = np.zeros_like(flat)
            test_indices = np.random.choice(flat.size, min(15, flat.size), replace=False)
            for ti in test_indices:
                old = flat[ti]
                flat[ti] = old + eps
                plus_loss = self.compute_loss(input_ids, targets)
                flat[ti] = old - eps
                minus_loss = self.compute_loss(input_ids, targets)
                flat[ti] = old
                grad_flat[ti] = (plus_loss - minus_loss) / (2 * eps)
            grads[name] = grad_flat.reshape(param.shape)
        for name, param in self._get_params():
            if name not in grads:
                grads[name] = np.zeros_like(param)
        return grads
    
    def update_weights(self, grads):
        self.t += 1
        for name, param in self._get_params():
            if name not in grads: continue
            g = grads[name] + self.weight_decay * param
            self.m[name] = 0.9 * self.m[name] + 0.1 * g
            self.v[name] = 0.999 * self.v[name] + 0.001 * (g ** 2)
            m_hat = self.m[name] / (1 - 0.9 ** self.t)
            v_hat = self.v[name] / (1 - 0.999 ** self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    
    def train(self, dataloader, epochs=10, num_grad_samples=30, verbose=True):
        history = []
        for epoch in range(epochs):
            total_loss = 0
            n = 0
            t0 = time.time()
            for input_ids, targets in dataloader:
                grads = self.compute_gradients(input_ids, targets, num_grad_samples)
                self.update_weights(grads)
                loss = self.compute_loss(input_ids, targets)
                total_loss += loss
                n += 1
            avg_loss = total_loss / max(n, 1)
            history.append(avg_loss)
            if verbose:
                print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} ({time.time()-t0:.1f}s)")
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
        return history
    
    def save_checkpoint(self, path, epoch=0):
        self.model.save(path)
        with open(os.path.join(path, "train_state.json"), "w") as f:
            json.dump({"epoch": epoch, "best_loss": self.best_loss, "step": self.step_count}, f)
    
    def load_checkpoint(self, path):
        self.model.load(path)
        state_path = os.path.join(path, "train_state.json")
        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
            self.best_loss = state.get("best_loss", float('inf'))
