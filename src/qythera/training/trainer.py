"""Complete training loop for the Qythera framework."""

import json
import math
import os
import pickle
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

from qythera.tensor import Tensor, no_grad
from qythera.nn import Module
from qythera import optim


class EMAModel:
    def __init__(self, model: Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: OrderedDict[str, np.ndarray] = OrderedDict()
        self.backup: OrderedDict[str, np.ndarray] = OrderedDict()
        for name, param in model.named_parameters():
            self.shadow[name] = param.data.copy()

    def update(self, model: Module):
        for name, param in model.named_parameters():
            self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self, model: Module):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.copy()
                param.data = self.shadow[name].copy()

    def restore(self, model: Module):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name].copy()
        self.backup.clear()

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": {k: v.copy() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: dict):
        self.decay = state["decay"]
        self.shadow = {k: v.copy() for k, v in state["shadow"].items()}


class GradScaler:
    def __init__(self, init_scale: float = 2**16, growth_factor: float = 2.0,
                 backoff_factor: float = 0.5, growth_interval: int = 2000):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.steps_since_last_scale = 0

    def scale_loss(self, loss: Tensor) -> Tensor:
        return loss * Tensor(np.array(self.scale))

    def step(self, optimizer: 'optim.Optimizer'):
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    g = p.grad.data if isinstance(p.grad, Tensor) else p.grad
                    p.grad = Tensor(g / self.scale)
        optimizer.step()
        self.steps_since_last_scale += 1
        if self.steps_since_last_scale >= self.growth_interval:
            self.scale *= self.growth_factor
            self.steps_since_last_scale = 0

    def unscale(self, optimizer: 'optim.Optimizer'):
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    g = p.grad.data if isinstance(p.grad, Tensor) else p.grad
                    p.grad = Tensor(g / self.scale)

    def state_dict(self) -> dict:
        return {"scale": self.scale, "steps_since_last_scale": self.steps_since_last_scale}

    def load_state_dict(self, state: dict):
        self.scale = state["scale"]
        self.steps_since_last_scale = state["steps_since_last_scale"]


class LabelSmoothingLoss:
    def __init__(self, vocab_size: int, smoothing: float = 0.1, ignore_index: int = -100):
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def __call__(self, logits: Tensor, target: Tensor) -> Tensor:
        if logits.ndim == 2:
            logits = logits.unsqueeze(0)
        if target.ndim == 1:
            target = target.unsqueeze(0)

        batch_size, seq_len, vocab_size = logits.shape
        log_probs = logits.log_softmax(axis=-1)
        target_np = target.numpy()

        smooth_val = self.smoothing / max(self.vocab_size - 1, 1)
        with no_grad():
            smooth_target = np.full((batch_size, seq_len, self.vocab_size), smooth_val, dtype=np.float32)
            valid = target_np != self.ignore_index
            safe_target = np.where(valid, target_np, 0)
            for b in range(batch_size):
                for s in range(seq_len):
                    if valid[b, s]:
                        smooth_target[b, s, :] = smooth_val
                        smooth_target[b, s, int(safe_target[b, s])] = 1.0 - self.smoothing
            smooth_target = Tensor(smooth_target)

        loss_per_token = -(smooth_target * log_probs).sum(axis=-1)
        mask_data = valid.astype(np.float32)
        valid_count = max(float(mask_data.sum()), 1e-8)
        loss = (loss_per_token * Tensor(mask_data)).sum() * (1.0 / valid_count)
        return loss


class StochasticDepth:
    def __init__(self, drop_prob: float = 0.0):
        self.drop_prob = drop_prob
        self.training = True

    def __call__(self, x: Tensor, residual: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x + residual
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = Tensor((np.random.rand(*shape) + keep_prob).astype(np.float32))
        mask = (random_tensor.floor() * (1.0 / keep_prob))
        return (x * mask) + residual


class GradNormHistory:
    def __init__(self, window: int = 100):
        self.history: List[float] = []
        self.window = window

    def update(self, norm: float):
        self.history.append(norm)
        if len(self.history) > self.window:
            self.history.pop(0)

    def is_anomaly(self, norm: float) -> bool:
        if len(self.history) < self.window:
            return False
        mean = np.mean(self.history)
        std = np.std(self.history)
        return abs(norm - mean) > 3 * std


class Trainer:
    def __init__(self, model: Module, optimizer: 'optim.Optimizer',
                 scheduler: Optional[Any] = None,
                 gradient_accumulation_steps: int = 1,
                 max_grad_norm: float = 1.0,
                 gradient_noise: float = 0.0,
                 batch_size_warmup_steps: int = 0,
                 initial_batch_size: int = 32,
                 loss_spike_threshold: float = 2.0,
                 loss_spike_window: int = 100,
                 loss_spike_lr_factor: float = 0.8,
                 ema_decay: float = 0.999,
                 use_ema: bool = True,
                 mixed_precision: bool = False,
                 label_smoothing: float = 0.0,
                 vocab_size: int = 32000,
                 label_smoothing_ignore_index: int = -100,
                 stochastic_depth_prob: float = 0.0,
                 weight_decay: float = 0.0,
                 checkpoint_dir: str = "checkpoints",
                 checkpoint_every: int = 1000,
                 checkpoint_keep: int = 5,
                 eval_every: int = 500,
                 log_every: int = 10,
                 device: str = "cpu"):
        self.model = model
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.model.train()
        self.weight_decay = weight_decay
        self.gradient_noise = gradient_noise
        self.batch_size_warmup_steps = batch_size_warmup_steps
        self.initial_batch_size = initial_batch_size
        self.optimizer = optimizer
        if self.weight_decay > 0:
            no_decay = ['bias', 'norm']
            decay_params = [p for n, p in self.model.named_parameters()
                            if p.requires_grad and not any(nd in n for nd in no_decay)]
            no_decay_params = [p for n, p in self.model.named_parameters()
                               if p.requires_grad and any(nd in n for nd in no_decay)]
            self.optimizer.param_groups.clear()
            self.optimizer.param_groups.append({
                'params': decay_params,
                **self.optimizer.defaults,
                'weight_decay': self.weight_decay
            })
            self.optimizer.param_groups.append({
                'params': no_decay_params,
                **self.optimizer.defaults,
                'weight_decay': 0.0
            })
        else:
            self.optimizer.param_groups[0]["params"] = [p for p in self.model.parameters() if p.requires_grad]
        self.scheduler = scheduler
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.loss_spike_threshold = loss_spike_threshold
        self.loss_spike_window = loss_spike_window
        self.loss_spike_lr_factor = loss_spike_lr_factor
        self.ema_decay = ema_decay
        self.use_ema = use_ema
        self.mixed_precision = mixed_precision
        self.label_smoothing = label_smoothing
        self.vocab_size = vocab_size
        self.label_smoothing_ignore_index = label_smoothing_ignore_index
        self.stochastic_depth_prob = stochastic_depth_prob
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every
        self.checkpoint_keep = checkpoint_keep
        self.eval_every = eval_every
        self.log_every = log_every
        self.device = device

        self.global_step = 0
        self.epoch = 0
        self.loss_history: List[float] = []
        self.running_loss = 0.0
        self.running_loss_count = 0
        self.tokens_processed = 0
        self.training_start_time = 0.0
        self.best_val_loss = float("inf")

        self.ema: Optional[EMAModel] = None
        if self.use_ema:
            self.ema = EMAModel(model, decay=self.ema_decay)

        self.grad_scaler: Optional[GradScaler] = None
        if self.mixed_precision:
            self.grad_scaler = GradScaler()

        self.label_smoothing_loss: Optional[LabelSmoothingLoss] = None
        if self.label_smoothing > 0:
            self.label_smoothing_loss = LabelSmoothingLoss(
                vocab_size=self.vocab_size,
                smoothing=self.label_smoothing,
                ignore_index=self.label_smoothing_ignore_index,
            )

        self.stochastic_depth: Optional[StochasticDepth] = None
        if self.stochastic_depth_prob > 0:
            self.stochastic_depth = StochasticDepth(drop_prob=self.stochastic_depth_prob)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _compute_grad_norm(self) -> float:
        params = [p for p in self.model.parameters() if p.grad is not None]
        if not params:
            return 0.0
        total = 0.0
        for p in params:
            g = p.grad.data if isinstance(p.grad, Tensor) else p.grad
            total += float(np.sum(g ** 2))
        return math.sqrt(total)

    def _has_nan_or_inf(self) -> List[str]:
        bad_params = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                g = param.grad.data if isinstance(param.grad, Tensor) else param.grad
                if np.any(np.isnan(g)) or np.any(np.isinf(g)):
                    bad_params.append(name)
        return bad_params

    def _clip_gradients(self):
        grad_norm = self._compute_grad_norm()
        if grad_norm > self.max_grad_norm:
            clip_coef = self.max_grad_norm / (grad_norm + 1e-8)
            for p in self.model.parameters():
                if p.grad is not None:
                    g = p.grad.data if isinstance(p.grad, Tensor) else p.grad
                    p.grad = Tensor(g * clip_coef)
        return grad_norm

    def _check_loss_spike(self, current_loss: float) -> bool:
        if len(self.loss_history) < self.loss_spike_window:
            return False
        window = self.loss_history[-self.loss_spike_window:]
        avg_loss = sum(window) / len(window)
        return current_loss > avg_loss * self.loss_spike_threshold

    def _reduce_lr(self):
        for group in self.optimizer.param_groups:
            group["lr"] *= self.loss_spike_lr_factor

    def _save_checkpoint(self, step: int, extra: Optional[dict] = None):
        ckpt = {
            "step": step,
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }
        if self.ema is not None:
            ckpt["ema_state_dict"] = self.ema.state_dict()
        if self.grad_scaler is not None:
            ckpt["grad_scaler_state_dict"] = self.grad_scaler.state_dict()
        if extra:
            ckpt.update(extra)

        path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{step}.pt")
        with open(path, "wb") as f:
            pickle.dump(ckpt, f)

        meta_path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{step}.json")
        with open(meta_path, "w") as f:
            json.dump({"step": step, "epoch": self.epoch, "loss": self.best_val_loss}, f)

        self._prune_checkpoints()

    def _prune_checkpoints(self):
        ckpt_files = []
        for fname in os.listdir(self.checkpoint_dir):
            if fname.startswith("checkpoint_step_") and fname.endswith(".pt"):
                try:
                    step_num = int(fname.split("_")[-1].replace(".pt", ""))
                    ckpt_files.append((step_num, fname))
                except ValueError:
                    continue
        ckpt_files.sort(key=lambda x: x[0])
        while len(ckpt_files) > self.checkpoint_keep:
            step_num, fname = ckpt_files.pop(0)
            pt_path = os.path.join(self.checkpoint_dir, fname)
            json_path = pt_path.replace(".pt", ".json")
            if os.path.exists(pt_path):
                os.remove(pt_path)
            if os.path.exists(json_path):
                os.remove(json_path)

    def _load_checkpoint(self, path: str):
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler and ckpt.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if self.ema and "ema_state_dict" in ckpt:
            self.ema.load_state_dict(ckpt["ema_state_dict"])
        if self.grad_scaler and "grad_scaler_state_dict" in ckpt:
            self.grad_scaler.load_state_dict(ckpt["grad_scaler_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.epoch = ckpt.get("epoch", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        return ckpt

    def load_last_checkpoint(self) -> Optional[str]:
        ckpt_files = []
        for fname in os.listdir(self.checkpoint_dir):
            if fname.startswith("checkpoint_step_") and fname.endswith(".pt"):
                try:
                    step_num = int(fname.split("_")[-1].replace(".pt", ""))
                    ckpt_files.append((step_num, os.path.join(self.checkpoint_dir, fname)))
                except ValueError:
                    continue
        if not ckpt_files:
            return None
        ckpt_files.sort(key=lambda x: x[0])
        path = ckpt_files[-1][1]
        self._load_checkpoint(path)
        return path

    @staticmethod
    def average_checkpoints(checkpoint_paths: List[str]) -> dict:
        if not checkpoint_paths:
            raise ValueError("No checkpoint paths provided")
        states = []
        for path in checkpoint_paths:
            with open(path, "rb") as f:
                states.append(pickle.load(f))

        avg_state = {}
        for key in states[0]["model_state_dict"]:
            stacked = np.stack([s["model_state_dict"][key] for s in states], axis=0)
            avg_state[key] = np.mean(stacked, axis=0)
        return avg_state

    def load_averaged_checkpoints(self, checkpoint_paths: List[str]):
        avg_state = self.average_checkpoints(checkpoint_paths)
        self.model.load_state_dict(avg_state)

    def _compute_loss(self, logits: Tensor, targets: Tensor,
                      loss_mask: Optional[Tensor] = None) -> Tensor:
        if self.label_smoothing_loss is not None:
            loss = self.label_smoothing_loss(logits, targets)
        else:
            if logits.ndim == 2:
                vocab_size = logits.shape[-1]
                logits_flat = logits
                targets_flat = targets
            else:
                vocab_size = logits.shape[-1]
                logits_flat = logits.reshape(-1, vocab_size)
                targets_flat = targets.reshape(-1)

            log_probs = logits_flat.log_softmax(axis=-1)
            with no_grad():
                target_indices = targets_flat.numpy().astype(int)
                safe_indices = np.clip(target_indices, 0, vocab_size - 1)
                n = targets_flat.shape[0]
                one_hot = Tensor(np.zeros((n, vocab_size), dtype=np.float32))
                one_hot.data[np.arange(n), safe_indices] = 1.0
                mask_data = (targets_flat.numpy() != self.label_smoothing_ignore_index).astype(np.float32)
                valid_count = max(float(mask_data.sum()), 1e-8)
            per_token_loss = -(one_hot * log_probs).sum(axis=-1)
            mask = Tensor(mask_data)
            loss = (per_token_loss * mask).sum() * (1.0 / valid_count)

        if loss_mask is not None:
            loss = loss * loss_mask.mean()
        return loss

    def _step(self, batch: dict, eval_fn: Optional[Callable] = None) -> dict:
        self.model.train()
        data = batch.get("input", batch.get("input_ids"))
        targets = batch.get("target", batch.get("labels"))
        loss_mask = batch.get("loss_mask", None)

        if isinstance(data, np.ndarray):
            data = Tensor(data)
        if isinstance(targets, np.ndarray):
            targets = Tensor(targets)
        if loss_mask is not None and isinstance(loss_mask, np.ndarray):
            loss_mask = Tensor(loss_mask)

        if self.batch_size_warmup_steps > 0 and self.global_step < self.batch_size_warmup_steps:
            current_batch = int(self.initial_batch_size +
                (data.shape[0] - self.initial_batch_size) * self.global_step / self.batch_size_warmup_steps)
            current_batch = max(1, min(current_batch, data.shape[0]))
            data = data[:current_batch]
            targets = targets[:current_batch]
            if loss_mask is not None:
                loss_mask = loss_mask[:current_batch]

        self.optimizer.zero_grad(set_to_none=True)

        if self.stochastic_depth is not None:
            self.stochastic_depth.training = True

        n_acc = self.gradient_accumulation_steps
        batch_size = data.shape[0]
        micro_batch_size = max(1, batch_size // n_acc) if n_acc > 1 else batch_size
        n_micro = (batch_size + micro_batch_size - 1) // micro_batch_size if n_acc > 1 else 1

        total_loss = 0.0
        for i in range(n_micro):
            start = i * micro_batch_size
            end = min(start + micro_batch_size, batch_size)
            dc = data[start:end]
            tc = targets[start:end]
            mc = loss_mask[start:end] if loss_mask is not None else None

            logits = self.model(dc)
            loss = self._compute_loss(logits, tc, mc)
            scaled_loss = self.grad_scaler.scale_loss(loss) if self.grad_scaler else loss
            scaled_loss = scaled_loss / n_micro
            scaled_loss.backward()
            total_loss += float(loss.data) / n_micro

        for p in self.model.parameters():
            if p.grad is not None and not isinstance(p.grad, Tensor):
                p.grad = Tensor(p.grad.copy())

        if not hasattr(self, 'grad_norms'):
            self.grad_norms: Dict[str, float] = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = np.linalg.norm(param.grad.data)
                self.grad_norms[name] = grad_norm

        step_log = {"loss": total_loss, "lr": self.optimizer.param_groups[0]["lr"]}

        self.tokens_processed += data.shape[0] * data.shape[1] if data.ndim >= 2 else data.shape[0]
        self.loss_history.append(total_loss)

        if self._check_loss_spike(total_loss):
            self.load_last_checkpoint()
            self._reduce_lr()
            step_log["loss_spike_detected"] = True
            self.model.train()
            return step_log

        bad_params = self._has_nan_or_inf()
        if bad_params:
            step_log["nan_inf_detected"] = bad_params
            return step_log

        grad_norm = self._clip_gradients()
        step_log["grad_norm"] = grad_norm

        if self.gradient_noise > 0:
            noise_std = self.gradient_noise / math.sqrt(1 + self.global_step)
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.data += np.random.normal(0, noise_std, p.grad.data.shape)

        if self.grad_scaler:
            self.grad_scaler.step(self.optimizer)
        else:
            self.optimizer.step()

        if self.ema is not None:
            self.ema.update(self.model)

        if self.scheduler is not None:
            self.scheduler.step()

        self.global_step += 1

        if self.global_step % self.checkpoint_every == 0:
            self._save_checkpoint(self.global_step)

        if self.eval_every and self.global_step % self.eval_every == 0 and eval_fn is not None:
            val_loss = self.evaluate(eval_fn)
            step_log["val_loss"] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(self.global_step, extra={"best": True})

        elapsed = time.time() - self.training_start_time
        tokens_per_sec = self.tokens_processed / max(elapsed, 1e-8)
        step_log["tokens_per_sec"] = tokens_per_sec

        return step_log

    def evaluate(self, eval_fn: Callable, use_ema: bool = True) -> float:
        self.model.eval()
        if use_ema and self.ema is not None:
            self.ema.apply_shadow(self.model)

        try:
            val_loss = eval_fn(self.model)
        finally:
            if use_ema and self.ema is not None:
                self.ema.restore(self.model)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
        return val_loss

    def fit(self, train_data: Iterator, val_data: Optional[Iterator] = None,
            epochs: int = 1, steps_per_epoch: Optional[int] = None,
            eval_fn: Optional[Callable] = None) -> List[dict]:
        self.training_start_time = time.time()
        logs: List[dict] = []

        for epoch in range(self.epoch, self.epoch + epochs):
            self.epoch = epoch
            step_in_epoch = 0

            for batch in train_data:
                if steps_per_epoch and step_in_epoch >= steps_per_epoch:
                    break

                log_entry = self._step(batch, eval_fn)
                log_entry["epoch"] = epoch
                log_entry["step"] = self.global_step
                log_entry["step_in_epoch"] = step_in_epoch
                logs.append(log_entry)

                if self.global_step % self.log_every == 0:
                    self._print_log(log_entry)

                step_in_epoch += 1

        return logs

    def _print_log(self, log_entry: dict):
        parts = [f"step={log_entry.get('step', 0)}"]
        if "loss" in log_entry:
            parts.append(f"loss={log_entry['loss']:.4f}")
        if "lr" in log_entry:
            parts.append(f"lr={log_entry['lr']:.2e}")
        if "grad_norm" in log_entry:
            parts.append(f"gnorm={log_entry['grad_norm']:.4f}")
        if "tokens_per_sec" in log_entry:
            parts.append(f"tok/s={log_entry['tokens_per_sec']:.0f}")
        if "val_loss" in log_entry:
            parts.append(f"val_loss={log_entry['val_loss']:.4f}")
        if log_entry.get("loss_spike_detected"):
            parts.append("[SPIKE RECOVERED]")
        if log_entry.get("nan_inf_detected"):
            names = ", ".join(log_entry["nan_inf_detected"][:3])
            parts.append(f"[NaN/Inf: {names}]")
        print(" | ".join(parts))
